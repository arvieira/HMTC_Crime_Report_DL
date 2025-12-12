from pydantic import BaseModel, Field


# Expected request template
class GenerationRequest(BaseModel):
    request_id: int = Field(description="Identificador da requisição para parear a resposta")
    prompt: list = Field(description="Lista de mensagens do prompt do usuário para resposta como chat")
    max_tokens: int = Field(description="Número máximo de tokens na resposta")
    temperature: float = Field(description="Temperatura a ser utilizada no modelo")
    model: str = Field(description="Nome do modelo que será utilizado para atender a requisição")
