# Build and Upload Commands

The following commands are used for building and uploading the project to PyPI.

## Testing
```bash
pytest -v
```

## Cleaning Previous Builds
```powershell
Remove-Item -Path "ai_stepper.egg-info" -Recurse -Force
Remove-Item -Path "dist" -Recurse -Force
```

## Building the Project
```bash
python -m build
```

## Uploading to PyPI
```bash
python -m twine upload dist/*
```
