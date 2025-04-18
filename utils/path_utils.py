import os
import sys

def setup_project_root():
    """
    Set up the project root path and add it to sys.path.
    This should be called at the start of any script that needs to import from the project.
    """
    # Get the absolute path of the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to get the project root
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    
    # Add project root to Python path if not already there
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    return project_root 