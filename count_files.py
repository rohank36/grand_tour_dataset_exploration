from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile  

api = HfApi()
items = api.list_repo_tree(
    repo_id="leggedrobotics/grand_tour_dataset",
    repo_type="dataset",
    recursive=True
)
total_files = sum(1 for x in items if isinstance(x, RepoFile))
print(total_files)
