from pydantic import BaseModel, Field
from enum import Enum

class EffectSlot(BaseModel):
    is_active: bool = Field(description="True se o efeito estiver ligado na corrente de sinal. False se estiver desligado.")
    description_or_name: str = Field(description="Descrição genérica (para mockup/web) OU nome exato do manual (para o patch final). Retorne 'Vazio' se inativo.")
    settings: str = Field(description="Configurações dos botões. Retorne 'N/A' se inativo.")


class ToneBlueprint(BaseModel):
    overall_vibe: str = Field(description="Uma frase resumindo o estilo geral do timbre.")
    preamp: EffectSlot
    distortion: EffectSlot
    amplifier: EffectSlot
    noise_reduction: EffectSlot
    cabinet: EffectSlot
    equalizer: EffectSlot
    modulation: EffectSlot
    delay: EffectSlot
    reverb: EffectSlot


class RouteChoice(str, Enum):
    AUDIO = "audio"
    MOCKUP = "mockup"
    WEB = "web"


class RouteDecision(BaseModel):
    route: RouteChoice = Field(description="A rota escolhida para processar o input do usuário.")
    clean_query: str = Field(description="A extração apenas da intenção do usuário, sem saudações.")

