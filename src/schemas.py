from pydantic import BaseModel, Field
from enum import Enum
from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

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


class IntentType(str, Enum):
    CREATE = "create"
    CHAT = "chat"
    QA = "qa"


class SubRouteType(str, Enum):
    WEB = "web"
    MOCKUP = "mockup"
    AUDIO = "audio"
    NONE = "none"


class RouterDecision(BaseModel):
    intent: IntentType = Field(description="A intenção principal.")
    sub_route: SubRouteType = Field(description="Se a intent for 'create', escolha 'web', 'mockup' ou 'audio'. Senão, use 'none'.")
    contextualized_query: str = Field(
        description="A reescrita da mensagem atual para que faça sentido absoluto isoladamente. Se for um pedido de TWEAK, inclua na reescrita a instrução para basear-se no patch atual."
    )


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: Optional[str]
    audio_path: Optional[str]
    is_audio: bool
    route: Optional[str]
    clean_query: Optional[str]
    blueprint: Optional[any]
    patch: Optional[any]
    agent_context: Optional[str]
    final_response: Optional[str]

