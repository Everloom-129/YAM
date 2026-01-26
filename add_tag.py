from huggingface_hub import HfApi

hub_api = HfApi()

hub_api.create_tag("williamtsai726/wipe_the_dish_0125", tag="v3.0", repo_type="dataset")

refs = hub_api.list_repo_refs("williamtsai726/wipe_the_dish_0125", repo_type="dataset")

# tags are under refs.tags
print([tag.name for tag in refs.tags])