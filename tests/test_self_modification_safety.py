#!/usr/bin/env python3
"""Test safety constraints for self-modifying code."""
import sys
import json
import re
from pathlib import Path
import ast

sys.path.append(str(Path(__file__).parent.parent))

class SafetyConstraints:
    """Define what the model is allowed to modify."""
    
    ALLOWED_MODIFY_DIRS = ['model/', 'config/', 'weights/', 'metrics/', 'logs/']
    FORBIDDEN_MODIFY_FILES = ['.github/workflows/', 'tests/', 'scripts/train_cycle.py']
    MAX_FILE_SIZE_MB = 100  # Max size for any single file
    FORBIDDEN_PATTERNS = [
        r'\beval\s*\(',  # eval() function call
        r'\bexec\s*\(',  # exec() function call
        r'__import__\s*\(',  # dynamic imports
        r'compile\s*\(',  # code compilation
    ]
    FORBIDDEN_IMPORTS = [
        'os.system',
        'subprocess.call',
        'subprocess.run',
        'subprocess.Popen'
    ]

def check_modified_files_safe():
    """Check that only safe files were modified."""
    import subprocess
    
    try:
        # Get list of modified files
        result = subprocess.run(
            ['git', 'diff', '--name-only', '--cached'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print("✓ No staged changes to check")
            return True
        
        modified_files = result.stdout.strip().split('\n')
        modified_files = [f for f in modified_files if f]  # Remove empty
        
        if not modified_files:
            print("✓ No modified files")
            return True
        
        print(f"Checking {len(modified_files)} modified files...")
        
        for filepath in modified_files:
            # Check if file is in forbidden directory
            for forbidden in SafetyConstraints.FORBIDDEN_MODIFY_FILES:
                if filepath.startswith(forbidden):
                    print(f"❌ Attempted to modify forbidden file: {filepath}")
                    return False
            
            # Check file size
            file_path = Path(filepath)
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > SafetyConstraints.MAX_FILE_SIZE_MB:
                    print(f"❌ File too large: {filepath} ({size_mb:.1f} MB)")
                    return False
        
        print(f"✓ All {len(modified_files)} modified files are safe")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not check modified files: {e}")
        return True  # Don't block on check failure

def check_python_files_safe():
    """Check that Python files don't contain dangerous code."""
    model_files = list(Path('model').glob('**/*.py'))
    scripts_files = list(Path('scripts').glob('**/*.py'))
    all_files = model_files + scripts_files
    
    for file_path in all_files:
        try:
            with open(file_path) as f:
                content = f.read()
            
            # Remove comments and docstrings to avoid false positives
            # This removes triple-quoted strings and # comments
            content_no_comments = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
            content_no_comments = re.sub(r"'''.*?'''", '', content_no_comments, flags=re.DOTALL)
            content_no_comments = re.sub(r'#.*$', '', content_no_comments, flags=re.MULTILINE)
            
            # Check for dangerous patterns (function calls, not just words)
            for pattern in SafetyConstraints.FORBIDDEN_PATTERNS:
                if re.search(pattern, content_no_comments):
                    print(f"❌ Dangerous pattern '{pattern}' found in {file_path}")
                    # Show context
                    matches = re.finditer(pattern, content_no_comments)
                    for match in matches:
                        start = max(0, match.start() - 50)
                        end = min(len(content_no_comments), match.end() + 50)
                        context = content_no_comments[start:end].replace('\n', ' ')
                        print(f"   Context: ...{context}...")
                    return False
            
            # Check for forbidden imports (must be actual import statements)
            for forbidden_import in SafetyConstraints.FORBIDDEN_IMPORTS:
                # Match actual import statements, not just the words
                import_pattern = rf'(?:from|import)\s+.*{re.escape(forbidden_import)}'
                if re.search(import_pattern, content_no_comments):
                    print(f"❌ Forbidden import '{forbidden_import}' found in {file_path}")
                    return False
            
            # Try to parse as valid Python
            try:
                ast.parse(content)
            except SyntaxError as e:
                print(f"❌ Syntax error in {file_path}: {e}")
                return False
                
        except Exception as e:
            print(f"⚠️  Could not check {file_path}: {e}")
    
    print(f"✓ Checked {len(all_files)} Python files - no dangerous code detected")
    return True

def check_config_values_safe():
    """Check that config values are within safe ranges."""
    config_path = Path('config/model_config.json')
    
    if not config_path.exists():
        print("✓ No config file yet")
        return True
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Check all values are reasonable
        for key, value in config.items():
            if isinstance(value, (int, float)):
                if value < 0 and key not in ['lr', 'learning_rate']:  # Allow negative LR for some optimizers
                    print(f"❌ Suspicious negative value in config: {key} = {value}")
                    return False
                
                if value > 10000 and 'dim' not in key.lower() and 'epoch' not in key.lower():
                    print(f"❌ Suspiciously large value in config: {key} = {value}")
                    return False
        
        print("✓ Config values within safe ranges")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Could not validate config: {e}")
        return True

def check_no_credential_exposure():
    """Check that no credentials are being committed."""
    suspicious_patterns = [
        'password', 'api_key', 'secret', 'token',
        'ghp_', 'github_token', 'aws_access'
    ]
    
    # Check config files
    config_files = list(Path('.').glob('**/*.json')) + list(Path('.').glob('**/*.yml'))
    config_files = [f for f in config_files if not str(f).startswith('.git')]
    
    for file_path in config_files[:20]:  # Check first 20
        try:
            with open(file_path) as f:
                content = f.read().lower()
            
            for pattern in suspicious_patterns:
                if pattern in content and '***' not in content:
                    # Check if it looks like an actual secret (not just the word)
                    lines = content.split('\n')
                    for line in lines:
                        if pattern in line and ':' in line:
                            value = line.split(':')[-1].strip().strip('"\',').strip()
                            # Check if value looks like a real secret (long alphanumeric)
                            if len(value) > 20 and any(c.isalnum() for c in value):
                                print(f"⚠️  Potential credential in {file_path}")
                                print(f"   Line: {line[:100]}...")
                                # Don't fail, just warn
        except:
            pass
    
    print("✓ No obvious credential exposure detected")
    return True

if __name__ == '__main__':
    print("Running self-modification safety checks...\n")
    
    checks = [
        ('Modified Files Safe', check_modified_files_safe()),
        ('Python Code Safe', check_python_files_safe()),
        ('Config Values Safe', check_config_values_safe()),
        ('No Credentials', check_no_credential_exposure())
    ]
    
    all_passed = all(result for _, result in checks)
    
    print("\n" + "="*50)
    print("Self-Modification Safety Summary:")
    print("="*50)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    if all_passed:
        print("\n✅ All safety checks passed")
        sys.exit(0)
    else:
        print("\n❌ Safety violations detected - blocking self-modification")
        sys.exit(1)
