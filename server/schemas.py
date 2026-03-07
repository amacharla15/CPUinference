#server/schemas.py

#This file should define the request shape.
#It exists so validation rules are centralized.
#Concept: schema = contract for incoming data
#Industry term: request schema, input validation, typed contract///


from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(gt=0, le=512)
    temperature: float = Field(ge=0.0, le=2.0)


class GenerationResponse(BaseModel):
    generated_text: str
    model_name: str