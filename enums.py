from enum import Enum


class RelationshipStrategy(Enum):
    Pairwise = "Pairwise"
    Parent = "Parent"
    Child = "Child"
    Confounder = "Confounder"
