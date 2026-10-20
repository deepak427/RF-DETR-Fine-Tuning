import os
from dotenv import dotenv_values
from roboflow import Roboflow

os.environ["ROBOFLOW_API_KEY"] = dotenv_values(".env")["ROBOFLOW_API_KEY"]

# Preflight
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
ws = rf.workspace("deepak-singh-1ojve")
print("Workspace OK")

# class_names.txt must be in the model_path folder
# This is what makes Roboflow use correct class labels
print("Uploading with class_names.txt from output/ folder...")
ws.deploy_model(
    model_type="rfdetr-medium",
    model_path=os.path.abspath("output"),
    project_ids=["ntlnp-suhy0"],
    version=2,
    filename="checkpoint_best_total.pth",
)
print("Done. Check app.roboflow.com/deepak-singh-1ojve/ntlnp-suhy0/models")
