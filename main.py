from pkg.plugin.context import register, handler, llm_func, BasePlugin, APIHost, EventContext
from pkg.plugin.events import PersonNormalMessageReceived, GroupNormalMessageReceived
from pkg.provider.modelmgr.requesters.chatcmpl import OpenAIChatCompletions
import typing
import json
from pkg.provider.entities import Message
import openai.types.chat.chat_completion as chat_completion

class DummyMessage:
    def __init__(self, content, role="assistant"):
        self.content = content
        self.role = role
    
    def dict(self):
        return {"content": self.content, "role": self.role}

class DummyChoice:
    def __init__(self, message):
        self.message = message

class DummyChatCompletion:
    def __init__(self, choices):
        self.choices = choices

@register(name="langbot流试传输兼容器", description="兼容流试传输", version="1.0", author="小馄饨")
class StreamHandler(BasePlugin):

    def __init__(self, host: APIHost):
        super().__init__(host)
        self._original_req = None
        self._requester = None

    async def initialize(self):
        pass

    def _process_stream_response(self, content: str) -> str:
        if not isinstance(content, str):
            return content

        if not content.startswith("data:"):
            return content

        try:
            parts = content.split("data:")
            combined_message = ""
            for part in parts:
                part = part.strip()
                if not part or part == "[DONE]":
                    continue

                try:
                    json_data = json.loads(part)
                    if isinstance(json_data, dict):
                        if "content" in json_data:
                            combined_message += json_data["content"]
                        elif "choices" in json_data and json_data["choices"]:
                            choice = json_data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                combined_message += choice["delta"]["content"]
                            elif "message" in choice and "content" in choice["message"]:
                                combined_message += choice["message"]["content"]
                except json.JSONDecodeError:
                    if '"content":"' in part:
                        start = part.find('"content":"') + 11
                        end = part.find('"', start)
                        if start > 10 and end > start:
                            combined_message += part[start:end]
                except Exception as e:
                    self.ap.logger.error(f"处理部分失败: {e}")
                    continue

            return combined_message if combined_message else content

        except Exception as e:
            self.ap.logger.error(f"处理流式响应失败: {str(e)}")
            return content

    @handler(PersonNormalMessageReceived)
    async def handle_person_message(self, ctx: EventContext):
        try:
            if not self._requester and hasattr(ctx.event.query, 'use_model'):
                requester = ctx.event.query.use_model.requester
                if isinstance(requester, OpenAIChatCompletions):
                    self._requester = requester
                    self._original_req = requester._req
                    requester._req = self._wrapped_req

            await self._process_messages(ctx.event.query)
        except Exception as e:
            self.ap.logger.error(f"处理个人消息失败: {str(e)}")

    async def _wrapped_req(self, args: dict) -> chat_completion.ChatCompletion:
        try:
            resp = await self._original_req(args)
            
            if isinstance(resp, str):
                processed_content = self._process_stream_response(resp)
                return chat_completion.ChatCompletion(
                    id="dummy_id",
                    choices=[
                        chat_completion.Choice(
                            index=0,
                            message=chat_completion.ChatCompletionMessage(
                                role="assistant",
                                content=processed_content
                            ),
                            finish_reason="stop"
                        )
                    ],
                    created=0,
                    model=args.get("model", "unknown"),
                    object="chat.completion",
                    system_fingerprint="dummy_fingerprint",
                    usage=chat_completion.CompletionUsage(
                        completion_tokens=0,
                        prompt_tokens=0,
                        total_tokens=0
                    )
                )
            return resp
        except Exception as e:
            self.ap.logger.error(f"处理请求失败: {str(e)}")
            raise

    async def _process_messages(self, query):
        if not query or not hasattr(query, 'resp_messages'):
            return

        for msg in query.resp_messages:
            if not isinstance(msg, Message) or not isinstance(msg.content, str):
                continue
                
            try:
                self.ap.logger.info(f"处理消息: {msg.content[:100]}...")
                processed_content = self._process_stream_response(msg.content)
                if processed_content:
                    msg.content = processed_content
                    self.ap.logger.info(f"处理后: {msg.content[:100]}...")
            except Exception as e:
                self.ap.logger.error(f"处理消息失败: {str(e)}")

    @handler(GroupNormalMessageReceived)
    async def handle_group_message(self, ctx: EventContext):
        try:
            if not self._requester and hasattr(ctx.event.query, 'use_model'):
                requester = ctx.event.query.use_model.requester
                if isinstance(requester, OpenAIChatCompletions):
                    self._requester = requester
                    self._original_req = requester._req
                    requester._req = self._wrapped_req

            await self._process_messages(ctx.event.query)
        except Exception as e:
            self.ap.logger.error(f"处理群组消息失败: {str(e)}")

    def __del__(self):
        if self._original_req and self._requester:
            try:
                self._requester._req = self._original_req
            except:
                pass