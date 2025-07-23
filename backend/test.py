from auth import get_access_token
from github import Github, Auth

auth = Auth.Token(get_access_token('BENCH'))
gh = Github(auth=auth)
gh.get_repo('suryajasper/github-api-test')