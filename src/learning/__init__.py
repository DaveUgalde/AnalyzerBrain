"""
Módulo de aprendizaje - Sistema de mejora continua y aprendizaje incremental.
Responsable de aprender de interacciones, refinar conocimiento y adaptarse a nuevos contextos.
"""

from .feedback_loop import FeedbackLoop
from .incremental_learner import IncrementalLearner
from .reinforcement_learner import ReinforcementLearner
from .knowledge_refiner import KnowledgeRefiner
from .adaptation_engine import AdaptationEngine
from .forgetting_mechanism import ForgettingMechanism
from .learning_evaluator import LearningEvaluator

__all__ = [
    'FeedbackLoop',
    'IncrementalLearner', 
    'ReinforcementLearner',
    'KnowledgeRefiner',
    'AdaptationEngine',
    'ForgettingMechanism',
    'LearningEvaluator'
]

__version__ = "1.0.0"
__author__ = "Project Brain Team"
__description__ = "Sistema de aprendizaje incremental y mejora continua"
