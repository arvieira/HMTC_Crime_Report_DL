from pydantic import BaseModel, Field


# Structured output for text clasfication
class CrimeType(BaseModel):
    delito: str = Field(description="Classificação do tipo do delito")
