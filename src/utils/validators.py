import re
from typing import List, Dict, Optional, Tuple, Union, Any

def validate_nucleotide_sequence(sequence: str) -> bool:
    """
    Validate if a sequence consists of valid nucleotides (A, T, G, C, U).
    
    Parameters
    ----------
    sequence : str
        Nucleotide sequence to validate
        
    Returns
    -------
    bool
        True if the sequence is valid, False otherwise
    """
    if not sequence:
        return False
    
    # Convert to uppercase for consistency
    sequence = sequence.upper()
    
    # Check if all characters are valid nucleotides
    valid_nucleotides = set("ATGCU")
    return all(nucleotide in valid_nucleotides for nucleotide in sequence)

def validate_dot_bracket_notation(structure: str) -> bool:
    """
    Validate if a structure in dot-bracket notation is well-formed.
    
    Parameters
    ----------
    structure : str
        Structure in dot-bracket notation
        
    Returns
    -------
    bool
        True if the structure is well-formed, False otherwise
    """
    if not structure:
        return False
    
    # Check for valid characters
    if not all(char in ".()[]{}" for char in structure):
        return False
    
    # Check for balanced brackets
    stack = []
    bracket_pairs = {')': '(', ']': '[', '}': '{'}
    
    for char in structure:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != bracket_pairs[char]:
                return False
    
    return len(stack) == 0

def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is well-formed.
    This is a basic validation and does not check for chemical validity.
    
    Parameters
    ----------
    smiles : str
        SMILES string to validate
        
    Returns
    -------
    bool
        True if the SMILES string is well-formed, False otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    # Check for balanced brackets and basic syntax
    bracket_count = 0
    
    for char in smiles:
        if char == '(':
            bracket_count += 1
        elif char == ')':
            bracket_count -= 1
            
        # A negative bracket count indicates unbalanced brackets
        if bracket_count < 0:
            return False
    
    # Check if all brackets are balanced
    if bracket_count != 0:
        return False
    
    # Check for common invalid patterns
    invalid_patterns = [
        r'[^A-Za-z0-9\(\)\[\]\{\}\.\+\-=#\$\:\\\/\*@\?,]',  # Invalid characters
        r'[A-Z][A-Z][A-Z]',  # Three uppercase letters (likely invalid element)
        r'\(\)',  # Empty brackets
        r'\[\]'   # Empty square brackets
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, smiles):
            return False
    
    return True

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration parameters.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    Tuple[bool, List[str]]
        (Valid, List of error messages)
    """
    errors = []
    
    # Check required top-level sections
    required_sections = ['data_processing', 'feature_extraction', 'structure_prediction', 
                         'modeling', 'aptamer_selection', 'targets', 'output']
    
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")
    
    # Check for at least one target
    if 'targets' in config and (not config['targets'] or len(config['targets']) == 0):
        errors.append("At least one target molecule must be specified")
    
    # Validate each target if present
    if 'targets' in config and isinstance(config['targets'], list):
        for i, target in enumerate(config['targets']):
            if not isinstance(target, dict):
                errors.append(f"Target at index {i} must be a dictionary")
                continue
                
            if 'name' not in target:
                errors.append(f"Target at index {i} is missing required 'name' field")
                
            if 'smiles' in target and not validate_smiles(target['smiles']):
                errors.append(f"Target '{target.get('name', f'at index {i}')}' has invalid SMILES: {target.get('smiles', '')}")
    
    # Check modeling parameters
    if 'modeling' in config:
        if 'binding_affinity' in config['modeling']:
            model_type = config['modeling']['binding_affinity'].get('model_type')
            if model_type not in ['xgboost', 'random_forest', 'neural_network', 'gradient_boosting']:
                errors.append(f"Invalid binding_affinity model_type: {model_type}")
    
    return len(errors) == 0, errors
