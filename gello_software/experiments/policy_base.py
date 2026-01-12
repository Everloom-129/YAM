from abc import ABC
from typing import Dict, Any, Union
import numpy as np


class PolicyBase(ABC):
    """
    Abstract base class for all policy implementations.
    
    This class defines the common interface that all policies must implement.
    Policies are responsible for taking observations and producing actions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the policy with configuration.
        
        Args:
            config: Configuration dictionary containing policy parameters
        """
        self.config = config
    
    def get_action_horizon(self):
        """
        The action horizon of the policy.
        """
        pass
    
    def prepare_input(self, obs: Dict[str, Any], instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Prepare the input for the policy.
        """
        pass
    
    
    def prepare_output(self, raw_actions):
        """
        Prepare the output for the policy.
        """
        return raw_actions
    
    def inference(self, obs: Dict[str, Any], **kwargs) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Inference the policy.
        """
        pass
    


