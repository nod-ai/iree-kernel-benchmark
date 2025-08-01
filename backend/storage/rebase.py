from auth import get_azure_clients, get_repo
from storage.conversion import parse_modification


def rebase_all(limit=40):
    db_client, dir_client = get_azure_clients()
    db_client.clear_all_repos()

    wave_repo = get_repo("wave")

    open_prs = wave_repo.get_pulls(state="all", sort="created", direction="desc")

    i = 0
    for gh_pr in open_prs:
        pr_dict = gh_pr.raw_data
        pr, merge = parse_modification(pr_dict)

        db_client.insert_pull_request(pr)
        if merge:
            db_client.insert_merge(merge)

        i += 1
        if i >= limit:
            return
