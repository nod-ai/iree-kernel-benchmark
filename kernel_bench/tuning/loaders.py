import json
import os
from kernel_bench.utils.print_utils import get_logger

logger = get_logger()


def parse_tuned_result(result_obj: dict):
    if "hyperparams" in result_obj:
        if "speedup" in result_obj and "improvement" in result_obj:
            if result_obj["speedup"] > 1 and result_obj["improvement"]:
                return result_obj["hyperparams"]
            else:
                return None
        return result_obj["hyperparams"]
    if "tuningConfig" in result_obj:
        return result_obj["tuningConfig"]
    return None


def load_tuning_configs_from_json(json_path: os.PathLike):
    try:
        with open(json_path, "r") as file:
            tuned_data = json.load(file)
    except:
        logger.error(f"Failed to parse tuned results file {json_path}")
        return {}

    if isinstance(tuned_data, dict):
        tuned_results = {
            name: parse_tuned_result(result) for name, result in tuned_data.items()
        }
    elif isinstance(tuned_data, list):
        tuned_results = {
            result["name"]: parse_tuned_result(result) for result in tuned_data
        }
    else:
        logger.error(f"Invalid format for tuned results")
        return {}

    print(tuned_results)
    return tuned_results
