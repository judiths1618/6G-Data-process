"""
Unified Directory Structure Manager

Ensures consistent structure across:
- saved_models/     (trained models)
- generated/        (synthesis outputs)
- logs/            (training logs)
- results/         (evaluation results)

All based on the same prepared_dir path.

Example:
  prepared_dir: ./work/EUR/prepared_rabbitmq
  
  Generated structure:
    saved_models/EUR/rabbitmq/
    generated/EUR/rabbitmq/
    logs/EUR/rabbitmq/
    results/EUR/rabbitmq/
"""

from pathlib import Path
from typing import Dict

class DirectoryManager:
    """
    Unified directory structure manager for WaveStitch pipeline
    
    Usage:
        dm = DirectoryManager(prepared_dir="./work/EUR/prepared_rabbitmq")
        
        # Get paths
        model_dir = dm.get_path('saved_models')
        output_dir = dm.get_path('generated')
        log_dir = dm.get_path('logs')
        
        # Or get all at once
        paths = dm.get_all_paths()
    """
    
    def __init__(self, prepared_dir: str):
        """
        Initialize directory manager
        
        Args:
            prepared_dir: Path to prepared data (e.g., "./work/EUR/prepared_rabbitmq")
        """
        self.prepared_dir = Path(prepared_dir)
        self.dataset_name, self.region = self._parse_prepared_dir()
        
    def _parse_prepared_dir(self):
        """
        Parse prepared_dir to extract dataset name and region
        
        Returns:
            (dataset_name, region) tuple
            
        Examples:
            "./work/EUR/prepared_rabbitmq" -> ("rabbitmq", "EUR")
            "prepared_amf" -> ("amf", None)
            "/path/to/USA/prepared_sensor1" -> ("sensor1", "USA")
        """
        path = self.prepared_dir.resolve()
        
        # Get last directory name
        last_dir = path.name
        
        # Extract dataset name (remove "prepared_" prefix)
        if last_dir.startswith("prepared_"):
            dataset_name = last_dir[len("prepared_"):]
        else:
            dataset_name = last_dir
        
        # Get parent directory (region/category)
        parent = path.parent.name
        
        # Check if parent is meaningful (not generic like "work", "data")
        generic_names = [".", "..", "work", "data", "datasets", "prepared", "experiments"]
        if parent and parent.lower() not in generic_names:
            region = parent
        else:
            region = None
        
        return dataset_name, region
    
    def get_path(self, base_dir: str, create: bool = False) -> Path:
        """
        Get path for a specific base directory
        
        Args:
            base_dir: Base directory name ("saved_models", "generated", "logs", etc.)
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Full path to the directory
        """
        if self.region:
            # Include region: base_dir/region/dataset
            full_path = Path(base_dir) / self.region / self.dataset_name
        else:
            # No region: base_dir/dataset
            full_path = Path(base_dir) / self.dataset_name
        
        if create:
            full_path.mkdir(parents=True, exist_ok=True)
        
        return full_path
    
    def get_all_paths(self, create: bool = False) -> Dict[str, Path]:
        """
        Get all standard paths
        
        Args:
            create: Whether to create directories
            
        Returns:
            Dictionary with keys: saved_models, generated, logs, results
        """
        bases = ["saved_models", "generated", "logs", "results"]
        paths = {base: self.get_path(base, create=create) for base in bases}
        return paths
    
    def get_model_path(self, epoch: int = None, final: bool = False) -> Path:
        """Get path for model checkpoint"""
        model_dir = self.get_path("saved_models", create=True)
        
        if final:
            return model_dir / "final_model.pt"
        elif epoch is not None:
            return model_dir / f"model_epoch_{epoch}.pt"
        else:
            return model_dir / "model.pt"
    
    def get_output_path(self, filename: str) -> Path:
        """Get path for generated output file"""
        output_dir = self.get_path("generated", create=True)
        return output_dir / filename
    
    def get_log_path(self, filename: str = "training.log") -> Path:
        """Get path for log file"""
        log_dir = self.get_path("logs", create=True)
        return log_dir / filename
    
    def get_result_path(self, filename: str) -> Path:
        """Get path for evaluation results"""
        result_dir = self.get_path("results", create=True)
        return result_dir / filename
    
    def print_structure(self):
        """Print the directory structure"""
        print("="*70)
        print("DIRECTORY STRUCTURE")
        print("="*70)
        print(f"\nPrepared dir: {self.prepared_dir}")
        print(f"Dataset: {self.dataset_name}")
        if self.region:
            print(f"Region: {self.region}")
        
        print(f"\nGenerated structure:")
        paths = self.get_all_paths(create=False)
        for name, path in paths.items():
            print(f"  {name:15s} -> {path}")
    
    def __repr__(self):
        return f"DirectoryManager(dataset={self.dataset_name}, region={self.region})"


# Convenience functions for backward compatibility
def get_generated_root(prepared_dir: str) -> str:
    """Sibling ``generated`` directory for a prepared bundle.

    Unifies the two bundle-naming conventions onto a single output tree so that
    model checkpoints, imputed CSVs, and evaluation artifacts all co-locate:

        DataOps:      <name>_regularized  ->  <name>_generated
        experiments:  prepared_<subset>   ->  generated_<subset>
        fallback:     <name>              ->  generated_<name>

    The result is a sibling of ``prepared_dir`` (derived from its actual
    location, so it is CWD-independent). ``scripts/reproduce_all.sh`` writes its
    ``--output-dir`` to the DataOps ``<name>_generated`` path, so resolving the
    same directory here keeps checkpoints in that one tree instead of a stray
    ``generated_<name>_regularized`` folder.
    """
    p = Path(prepared_dir)
    name = p.name
    if name.endswith("_regularized"):
        gen = name[: -len("_regularized")] + "_generated"
    elif name.startswith("prepared_"):
        gen = "generated_" + name[len("prepared_"):]
    else:
        gen = f"generated_{name}"
    return str(p.parent / gen)


def get_save_dir(prepared_dir: str) -> str:
    """Model checkpoint directory, co-located UNDER the subset's generated folder.

    Returns ``<get_generated_root>/saved_models`` (see :func:`get_generated_root`
    for the naming rules). Train (save) and synthesis (load) both call this, so
    they always agree. The container flow (run_pipeline.py) manages its own
    ``prepared/saved_model`` path and is unaffected.
    """
    return str(Path(get_generated_root(prepared_dir)) / "saved_models")

def get_generated_dir(prepared_dir: str) -> str:
    """Get generated directory path"""
    dm = DirectoryManager(prepared_dir)
    return str(dm.get_path("generated"))

def get_log_dir(prepared_dir: str) -> str:
    """Get logs directory path"""
    dm = DirectoryManager(prepared_dir)
    return str(dm.get_path("logs"))

def get_all_dirs(prepared_dir: str) -> Dict[str, str]:
    """Get all directory paths as strings"""
    dm = DirectoryManager(prepared_dir)
    paths = dm.get_all_paths()
    return {k: str(v) for k, v in paths.items()}


# Example usage and tests
if __name__ == '__main__':
    # Test cases
    test_cases = [
        "./work/EUR/prepared_rabbitmq",
        "prepared_amf",
        "/path/to/USA/prepared_sensor1",
        "./data/prepared_deepsense",
    ]
    
    print("="*70)
    print("DIRECTORY STRUCTURE TESTS")
    print("="*70)
    
    for prepared_dir in test_cases:
        print(f"\n{'-'*70}")
        dm = DirectoryManager(prepared_dir)
        dm.print_structure()
        
        # Example file paths
        print(f"\nExample paths:")
        print(f"  Model (epoch 10): {dm.get_model_path(epoch=10)}")
        print(f"  Model (final):    {dm.get_model_path(final=True)}")
        print(f"  Output:           {dm.get_output_path('predictions.csv')}")
        print(f"  Log:              {dm.get_log_path()}")
        print(f"  Results:          {dm.get_result_path('metrics.json')}")
    
    # Test convenience functions
    print(f"\n{'='*70}")
    print("CONVENIENCE FUNCTIONS")
    print("="*70)
    
    prepared_dir = "./work/EUR/prepared_rabbitmq"
    print(f"\nInput: {prepared_dir}")
    print(f"  save_dir:      {get_save_dir(prepared_dir)}")
    print(f"  generated_dir: {get_generated_dir(prepared_dir)}")
    print(f"  log_dir:       {get_log_dir(prepared_dir)}")
    
    all_dirs = get_all_dirs(prepared_dir)
    print(f"\nAll dirs:")
    for name, path in all_dirs.items():
        print(f"  {name}: {path}")