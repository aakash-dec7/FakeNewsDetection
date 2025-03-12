from pathlib import Path


project_name = "fnd"

list_of_files = [
    ".github/workflows/main.yaml",
    "src/__init__.py",
    f"src/{project_name}/component/__init__.py",
    f"src/{project_name}/component/s1_data_ingestion.py",
    f"src/{project_name}/component/s2_data_validation.py",
    f"src/{project_name}/component/s3_data_preprocessing.py",
    f"src/{project_name}/component/s4_data_transformation.py",
    f"src/{project_name}/component/s5_model_training.py",
    f"src/{project_name}/component/s6_model_evaluation.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/entity.py",
    f"src/{project_name}/prediction/__init__.py",
    f"src/{project_name}/prediction/prediction.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/utils.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/constant/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    "config/config.yaml",
    "config/params.yaml",
    "config/schema.yaml",
    "templates/index.html",
    "app.py",
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "deployment.yaml",
    "setup.py",
    "main.py",
    "test.py",
    "dvc.yaml",
]


for filepath in map(Path, list_of_files):
    filedir = filepath.parent  # Get the directory path
    filename = filepath.name  # Get the file name

    # Create the directory if it doesn't exist
    if filedir and not filedir.exists():
        filedir.mkdir(parents=True, exist_ok=True)

    # Create the file if it doesn't exist
    if not filepath.exists():
        with open(filepath, "w") as f:
            pass  # Creates an empty file without modifying existing onesS
    else:
        print(f"File already exists: {filepath}")
