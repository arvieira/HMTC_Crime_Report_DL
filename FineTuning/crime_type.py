from pydantic import BaseModel, Field

class CrimeType(BaseModel):
    delito: str = Field(description="Crime type classification")