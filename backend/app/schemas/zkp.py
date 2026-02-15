from typing import List, Union
from pydantic import BaseModel

class Proof(BaseModel):
    pi_a: List[str]
    pi_b: List[List[str]]
    pi_c: List[str]
    protocol: str = "groth16"
    curve: str = "bn128"

class ZKPayload(BaseModel):
    proof: Proof
    public_signals: List[str]
