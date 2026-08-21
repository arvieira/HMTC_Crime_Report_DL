from pydantic import BaseModel, Field


# Expected response template
class GenerationResponse(BaseModel):
    response_id: int = Field(description="Identificador da requisição para parear com a resposta")
    generated_text: str = Field(description="Texto da resposta do modelo")
    token_probabilities: list = Field(description="Vetor de probabilidades dos tokens da resposta")
    classification: str = Field(description="Classe atribuída ao prompt inicial")
    full_prob: float = Field(description="Probabilidade da linha delito e classe")
    full_logprob: float = Field(description="Logits da linha delito e classe")
    class_prob: float = Field(description="Probabilidade somente da classe")
    class_logprob: float = Field(description="Logits somente da classe")
