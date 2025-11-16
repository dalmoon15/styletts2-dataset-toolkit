#!/usr/bin/env python
"""
Automated installer for monotonic_align package with validation.

This script:
1. Clones the monotonic_align repository from GitHub
2. Installs it with pip (compiles Cython extensions)
3. Tests that it works correctly

Requirements:
- Git (for cloning)
- Microsoft C++ Build Tools (Windows, for Cython compilation)
- PyTorch (must be installed first)
"""
import os
import sys
import shutil
import subprocess
import tempfile
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monotonic_align_install.log')
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, cwd=None):
    """Run a command and return its output"""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd,
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        raise

def check_prerequisites():
    """Check if required tools are installed"""
    logger.info("Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error(f"Python 3.8+ required, found {sys.version}")
        return False
    logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        logger.error("PyTorch not found. Install it first!")
        return False
    
    # Check Git
    try:
        run_command(['git', '--version'])
        logger.info("✓ Git available")
    except Exception:
        logger.error("Git not found. Please install Git first.")
        return False
    
    # Check C++ compiler (Windows)
    if platform.system() == 'Windows':
        try:
            import distutils.ccompiler
            compiler = distutils.ccompiler.new_compiler()
            logger.info("✓ C++ compiler available")
        except Exception:
            logger.warning("C++ compiler may not be available")
            logger.warning("Install Microsoft C++ Build Tools if compilation fails")
    
    return True

def install_monotonic_align():
    """Install the monotonic_align package"""
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Temporary directory: {temp_dir}")
    
    try:
        # Clone repository
        logger.info("Cloning monotonic_align repository...")
        run_command(
            ['git', 'clone', 'https://github.com/resemble-ai/monotonic_align.git'],
            cwd=temp_dir
        )
        
        repo_dir = os.path.join(temp_dir, 'monotonic_align')
        
        # Install package
        logger.info("Installing monotonic_align (compiling Cython extensions)...")
        logger.info("This may take 1-2 minutes...")
        run_command(
            [sys.executable, '-m', 'pip', 'install', '-e', '.'],
            cwd=repo_dir
        )
        
        return True
    
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        return False
    
    finally:
        # Clean up
        try:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up {temp_dir}: {e}")

def test_installation():
    """Test that monotonic_align works correctly"""
    logger.info("Testing monotonic_align installation...")
    
    test_code = """
import torch
try:
    from monotonic_align import maximum_path
    from monotonic_align import mask_from_lens
    
    # Create test tensors
    batch_size = 2
    max_len = 10
    neg_cent = torch.randn(batch_size, max_len, max_len)
    mask = torch.ones_like(neg_cent)
    
    # Test mask_from_lens
    input_lengths = torch.tensor([8, 6])
    target_lengths = torch.tensor([7, 5])
    mask_test = mask_from_lens(neg_cent, input_lengths, target_lengths)
    
    # Test maximum_path
    path = maximum_path(neg_cent, mask_test)
    
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
"""
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            logger.info("✓ monotonic_align test passed")
            return True
        else:
            logger.error(f"Test failed:")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False

def main():
    """Main installation workflow"""
    print("="*70)
    print("monotonic_align Installation Script")
    print("="*70)
    print()
    
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed")
        print("Please install missing dependencies and try again")
        print("See monotonic_align_install.log for details")
        sys.exit(1)
    
    print("\n✓ Prerequisites OK")
    print("\nInstalling monotonic_align...")
    print("This will:")
    print("  1. Clone from GitHub")
    print("  2. Compile Cython extensions")
    print("  3. Install with pip")
    print("  4. Run validation tests")
    print()
    
    # Install
    if not install_monotonic_align():
        print("\n❌ Installation failed")
        print("Check monotonic_align_install.log for details")
        sys.exit(1)
    
    print("\n✓ Installation complete")
    
    # Test
    print("\nTesting installation...")
    if not test_installation():
        print("\n⚠️  Installation succeeded but tests failed")
        print("monotonic_align may not work correctly")
        print("Check monotonic_align_install.log for details")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ SUCCESS: monotonic_align installed and tested")
    print("="*70)
    print("\nYou can now run StyleTTS2 training:")
    print("  launchers\\run_finetune_safe.bat")
    print("  or: python StyleTTS2/train_finetune.py")
    print()

if __name__ == "__main__":
    main()
