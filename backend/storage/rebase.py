from tqdm import tqdm
from backend.runs import get_artifact_parser
from backend.runs.run_utils import parse_run_from_gh
from backend.runs.workflows import SUPPORTED_WORKFLOWS
from backend.github_utils import get_repo
from .conversion import parse_pr_obj
from .types import *


def rebase_runs(limit=10):
    wave = get_repo("bench")

    for workflow in SUPPORTED_WORKFLOWS:
        gh_runs = wave.get_workflow(workflow.filename).get_runs(status="completed")

        i = 0
        for gh_run in gh_runs:
            db_run = WorkflowRunDb.find_by_id(str(gh_run.id))
            if not db_run:
                try:
                    db_run = parse_run_from_gh(gh_run)
                except:
                    continue

            # if db_run.conclusion != "success":
            #     continue

            if not db_run.hasArtifact:
                artifact_parser = get_artifact_parser(workflow.run_type)

                for artifact in gh_run.get_artifacts():
                    success, _ = artifact_parser.parse_and_save_artifact(
                        artifact, db_run
                    )
                    if success:
                        db_run.hasArtifact = True
                        break

            WorkflowRunDb.upsert(db_run)

            i += 1
            if i >= limit:
                break


def rebase_pull_requests(limit=40):
    wave_repo = get_repo("wave")
    open_prs = wave_repo.get_pulls(state="all", sort="created", direction="desc")

    pbar = tqdm(total=limit, desc="Rebasing pull requests")

    i = 0
    for gh_pr in open_prs:
        pr_dict = gh_pr.raw_data
        pr = parse_pr_obj(pr_dict)
        RepoPullRequestDb.upsert(pr)

        i += 1
        pbar.update()
        if i >= limit:
            return

    pbar.close()


def rebase_all(mod_limit=40, perf_limit=20):
    rebase_pull_requests(mod_limit)
    rebase_runs(perf_limit)
