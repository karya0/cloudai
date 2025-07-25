# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import copy
import logging
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional
from unittest.mock import Mock

import toml
import yaml

from cloudai.core import (
    BaseInstaller,
    CloudAIGymEnv,
    Installable,
    Parser,
    Registry,
    Runner,
    System,
    Test,
    TestParser,
    TestScenario,
)
from cloudai.models.scenario import ReportConfig
from cloudai.parser import HOOK_ROOT
from cloudai.systems.slurm import SingleSbatchRunner, SlurmSystem
from cloudai.util import prepare_output_dir


def handle_install_and_uninstall(args: argparse.Namespace) -> int:
    """
    Manage the installation or uninstallation process for CloudAI.

    Based on user-specified mode, utilizing the Installer class.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    parser = Parser(args.system_config)
    system, tests, scenario = parser.parse(args.tests_dir, args.test_scenario)

    if args.output_dir:
        system.output_path = args.output_dir.absolute()
    system.update()
    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")

    installables, installer = prepare_installation(system, tests, scenario)

    rc = 0
    if args.mode == "install":
        all_installed = installer.is_installed(installables)
        if all_installed:
            logging.info(f"CloudAI is already installed into '{system.install_path}'.")
        else:
            logging.info("Not all components are ready")
            result = installer.install(installables)
            if result.success:
                logging.info(f"CloudAI is successfully installed into '{system.install_path.absolute()}'.")
            else:
                logging.error(result.message)
                rc = 1

    elif args.mode == "uninstall":
        logging.info("Uninstalling test templates.")
        result = installer.uninstall(installables)
        if result.success:
            logging.info("Uninstallation successful.")
        else:
            logging.error(result.message)
            rc = 1

    return rc


def prepare_installation(
    system: System, tests: list[Test], scenario: Optional[TestScenario]
) -> tuple[list[Installable], BaseInstaller]:
    installables: list[Installable] = []
    if scenario:
        for test in scenario.test_runs:
            logging.debug(f"{test.test.name} has {len(test.test.test_definition.installables)} installables.")
            installables.extend(test.test.test_definition.installables)
    else:
        for test in tests:
            logging.debug(f"{test.name} has {len(test.test_definition.installables)} installables.")
            installables.extend(test.test_definition.installables)

    registry = Registry()
    installer_class = registry.installers_map.get(system.scheduler)
    if installer_class is None:
        raise NotImplementedError(f"No installer available for scheduler: {system.scheduler}")
    installer = installer_class(system)

    return installables, installer


def handle_dse_job(runner: Runner, args: argparse.Namespace):
    registry = Registry()

    original_test_runs = copy.deepcopy(runner.runner.test_scenario.test_runs)

    for tr in runner.runner.test_scenario.test_runs:
        test_run = copy.deepcopy(tr)
        env = CloudAIGymEnv(test_run=test_run, runner=runner)
        agent_type = test_run.test.test_definition.agent

        agent_class = registry.agents_map.get(agent_type)
        if agent_class is None:
            logging.error(
                f"No agent available for type: {agent_type}. Please make sure {agent_type} "
                f"is a valid agent type. Available agents: {registry.agents_map.keys()}"
            )
            continue

        agent = agent_class(env)
        for step in range(agent.max_steps):
            result = agent.select_action()
            if result is None:
                break
            step, action = result
            env.test_run.step = step
            observation, reward, done, info = env.step(action)
            feedback = {"trial_index": step, "value": reward}
            agent.update_policy(feedback)
            logging.info(f"Step {step}: Observation: {observation}, Reward: {reward}")

    if args.mode == "run":
        runner.runner.test_scenario.test_runs = original_test_runs
        generate_reports(runner.runner.system, runner.runner.test_scenario, runner.runner.scenario_root)

    logging.info("All jobs are complete.")


def generate_reports(system: System, test_scenario: TestScenario, result_dir: Path) -> None:
    registry = Registry()
    for name, reporter_class in registry.scenario_reports.items():
        logging.debug(f"Generating report '{name}' ({reporter_class.__name__})")

        cfg = registry.report_configs.get(name, ReportConfig(enable=False))
        if isinstance(system, SlurmSystem) and system.reports and name in system.reports:
            cfg = system.reports[name]
        logging.debug(f"Report '{name}' config is: {cfg.model_dump_json(indent=None)}")

        if not cfg.enable:
            logging.debug(f"Skipping report {name} because it is disabled.")
            continue

        try:
            reporter = reporter_class(system, test_scenario, result_dir, cfg)
            reporter.generate()
        except Exception as e:
            logging.warning(f"Error generating report '{name}', see debug log for details")
            logging.debug(e, stack_info=True)


def handle_non_dse_job(runner: Runner, args: argparse.Namespace) -> None:
    asyncio.run(runner.run())

    logging.info(f"All test scenario results stored at: {runner.runner.scenario_root}")

    if args.mode == "run":
        generate_reports(runner.runner.system, runner.runner.test_scenario, runner.runner.scenario_root)

    logging.info("All jobs are complete.")


def register_signal_handlers(signal_handler: Callable) -> None:
    """Register signal handlers for handling termination-related signals."""
    signals = [
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGHUP,
        signal.SIGQUIT,
    ]
    for sig in signals:
        signal.signal(sig, signal_handler)


def handle_dry_run_and_run(args: argparse.Namespace) -> int:
    parser = Parser(args.system_config)
    system, tests, test_scenario = parser.parse(args.tests_dir, args.test_scenario)

    assert test_scenario is not None

    if args.output_dir:
        system.output_path = args.output_dir.absolute()

    if not prepare_output_dir(system.output_path):
        return 1
    if args.mode == "dry-run":
        system.monitor_interval = 1
    system.update()

    if args.single_sbatch:
        if not isinstance(system, SlurmSystem):
            logging.error("Single sbatch is only supported for Slurm systems.")
            return 1

        Registry().update_runner("slurm", SingleSbatchRunner)

    logging.info(f"System Name: {system.name}")
    logging.info(f"Scheduler: {system.scheduler}")
    logging.info(f"Test Scenario Name: {test_scenario.name}")

    logging.info("Checking if test templates are installed.")

    installables, installer = prepare_installation(system, tests, test_scenario)

    if args.enable_cache_without_check:
        result = installer.mark_as_installed(installables)
    else:
        result = installer.is_installed(installables)

    if args.mode == "run" and not result.success:
        logging.error("CloudAI has not been installed. Please run install mode first.")
        logging.error(result.message)
        return 1

    logging.info(test_scenario.pretty_print())

    runner = Runner(args.mode, system, test_scenario)
    register_signal_handlers(runner.cancel_on_signal)

    has_dse = any(tr.is_dse_job for tr in test_scenario.test_runs)
    if args.single_sbatch or not has_dse:  # in this mode cases are unrolled using grid search
        handle_non_dse_job(runner, args)
        return 0

    if all(tr.is_dse_job for tr in test_scenario.test_runs):
        handle_dse_job(runner, args)
    else:
        logging.error("Mixing DSE and non-DSE jobs is not allowed.")
        return 1

    return 0


def handle_generate_report(args: argparse.Namespace) -> int:
    """
    Generate a report based on the existing configuration and test results.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
    """
    parser = Parser(args.system_config)
    system, _, test_scenario = parser.parse(args.tests_dir, args.test_scenario)
    assert test_scenario is not None

    logging.info("Generating report based on system and test scenario")
    generate_reports(system, test_scenario, args.result_dir)

    logging.info("Report generation completed.")

    return 0


def expand_file_list(root: Path, glob: str = "*.toml") -> tuple[int, List[Path]]:
    if not root.exists():
        logging.error(f"{root} does not exist.")
        return (1, [])

    test_tomls = [root]
    if root.is_dir():
        test_tomls = list(root.glob(glob))
        if not test_tomls:
            logging.error(f"No TOMLs found in {root}")
            return (1, [])

    return (0, test_tomls)


@contextmanager
def _ensure_kube_config_exists(system_toml_path: Path, content: str):
    try:
        config_dict = toml.loads(content)
    except Exception as e:
        logging.error(f"Error parsing TOML file {system_toml_path}: {e}")
        raise

    kube_config_path_str = config_dict.get("kube_config_path")
    kube_config_path = Path(kube_config_path_str) if kube_config_path_str else Path.home() / ".kube" / "config"

    created_file = False
    created_dir = False

    if not kube_config_path.exists():
        logging.warning(f"Kube config file '{kube_config_path}' not found. Creating a dummy one.")
        if not kube_config_path.parent.exists():
            kube_config_path.parent.mkdir(parents=True, exist_ok=True)
            created_dir = True

        dummy_config = {
            "apiVersion": "v1",
            "kind": "Config",
            "preferences": {},
            "clusters": [{"name": "dummy-cluster", "cluster": {"server": "https://dummy-server"}}],
            "users": [{"name": "dummy-user", "user": {"token": "dummy-token"}}],
            "contexts": [{"name": "dummy-context", "context": {"cluster": "dummy-cluster", "user": "dummy-user"}}],
            "current-context": "dummy-context",
        }
        kube_config_path.write_text(yaml.dump(dummy_config))
        created_file = True
    else:
        logging.debug(f"Kube config '{kube_config_path}' already exists. Skipping creation.")

    try:
        yield kube_config_path
    finally:
        if created_file:
            try:
                kube_config_path.unlink()
                logging.debug(f"Deleted temporary kube config: {kube_config_path}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary kube config '{kube_config_path}': {e}")
        if created_dir:
            try:
                kube_config_path.parent.rmdir()
                logging.debug(f"Deleted kube config directory: {kube_config_path.parent}")
            except OSError:
                pass


def verify_system_configs(system_tomls: List[Path]) -> int:
    nfailed = 0

    for system_toml in system_tomls:
        logging.debug(f"Verifying System: {system_toml}...")
        content = system_toml.read_text()

        if 'scheduler = "kubernetes"' in content:
            try:
                with _ensure_kube_config_exists(system_toml, content):
                    Parser.parse_system(system_toml)
            except Exception:
                nfailed += 1
        else:
            try:
                Parser.parse_system(system_toml)
            except Exception:
                nfailed += 1

    if nfailed:
        logging.error(f"{nfailed} out of {len(system_tomls)} system configurations have issues.")
    else:
        logging.info(f"Checked systems: {len(system_tomls)}, all passed")

    return nfailed


def verify_test_configs(test_tomls: List[Path], strict: bool) -> int:
    nfailed = 0
    tp = TestParser([], None)  # type: ignore
    logging.info(f"Strict test verification: {strict}")
    for test_toml in test_tomls:
        logging.debug(f"Verifying Test: {test_toml}...")
        try:
            with test_toml.open() as fh:
                tp.current_file = test_toml
                tp.load_test_definition(toml.load(fh), strict)
        except Exception:
            nfailed += 1

    if nfailed:
        logging.error(f"{nfailed} out of {len(test_tomls)} test configurations have issues.")
    else:
        logging.info(f"Checked tests: {len(test_tomls)}, all passed")

    return nfailed


def verify_test_scenarios(
    scenario_tomls: List[Path],
    test_tomls: list[Path],
    hook_tomls: List[Path],
    hook_test_tomls: list[Path],
    strict: bool = False,
) -> int:
    system = Mock(spec=System)
    nfailed = 0
    for scenario_file in scenario_tomls:
        logging.debug(f"Verifying Test Scenario: {scenario_file}...")
        try:
            tests = Parser.parse_tests(test_tomls, system)
            hook_tests = Parser.parse_tests(hook_test_tomls, system)
            hooks = Parser.parse_hooks(hook_tomls, system, {t.name: t for t in hook_tests})
            Parser.parse_test_scenario(scenario_file, system, {t.name: t for t in tests}, hooks, strict)
        except Exception:
            nfailed += 1

    if nfailed:
        logging.error(f"{nfailed} out of {len(scenario_tomls)} test scenarios have issues.")
    else:
        logging.info(f"Checked scenarios: {len(scenario_tomls)}, all passed")

    return nfailed


def handle_verify_all_configs(args: argparse.Namespace) -> int:
    root: Path = args.configs_dir
    err, tomls = expand_file_list(root, glob="**/*.toml")
    if err:
        return err

    err, hook_tomls = expand_file_list(HOOK_ROOT, glob="**/*.toml")
    tomls += hook_tomls
    logging.info(f"Found {len(hook_tomls)} hook TOMLs (always verified)")

    files = load_tomls_by_type(tomls)

    test_tomls = files["test"]
    if args.tests_dir:
        test_tomls = list(args.tests_dir.glob("*.toml"))
    elif files["scenario"]:
        logging.warning(
            "Test configuration directory not provided, using all found test TOMLs in the specified directory."
        )

    nfailed = 0
    if files["system"]:
        nfailed += verify_system_configs(files["system"])
    if files["test"]:
        nfailed += verify_test_configs(files["test"], args.strict)
    if files["scenario"]:
        nfailed += verify_test_scenarios(files["scenario"], test_tomls, files["hook"], files["hook_test"], args.strict)
    if files["unknown"]:
        logging.error(f"Unknown configuration files: {[str(f) for f in files['unknown']]}")
        nfailed += len(files["unknown"])

    if nfailed:
        logging.error(f"{nfailed} out of {len(tomls)} configuration files have issues.")
    else:
        logging.info(f"Checked {len(tomls)} configuration files, all passed")

    return nfailed


def load_tomls_by_type(tomls: List[Path]) -> dict[str, List[Path]]:
    files: dict[str, List[Path]] = {
        "system": [],
        "test": [],
        "scenario": [],
        "hook_test": [],
        "hook": [],
        "unknown": [],
    }
    for toml_file in tomls:
        content = toml_file.read_text()

        is_in_hook_root = False
        try:
            toml_file.relative_to(HOOK_ROOT)
            is_in_hook_root = True
        except ValueError:
            pass

        if is_in_hook_root:
            if "test" in toml_file.parts:
                files["hook_test"].append(toml_file)
            else:
                files["hook"].append(toml_file)
            continue

        if "scheduler =" in content:
            files["system"].append(toml_file)
        elif "test_template_name =" in content and "[[Tests]]" not in content:
            files["test"].append(toml_file)
        elif "[[Tests]]" in content:
            files["scenario"].append(toml_file)
        else:
            files["unknown"].append(toml_file)

    return files


def handle_list_registered_items(args: argparse.Namespace) -> int:
    item_type = args.type
    registry = Registry()
    if item_type == "reports":
        print("Registered scenario reports:")
        for idx, (name, report) in enumerate(sorted(registry.scenario_reports.items()), start=1):
            str = f'{idx}. "{name}" {report.__name__}'
            if args.verbose:
                str += f" (config={registry.report_configs[name].model_dump_json(indent=None)})"
            print(str)

    return 0
