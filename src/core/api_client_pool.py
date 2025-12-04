# src/core/api_client_pool.py
import os
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from dotenv import load_dotenv
import logging
from typing import List, Optional

load_dotenv()

class OpenAIClientPool:
    def __init__(self):
        keys_str = os.getenv('OPENAI_API_KEYS', '')
        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1').rstrip('/')
        model = os.getenv('OPENAI_MODEL_NAME', 'deepseek-ai/DeepSeek-V3')

        if not keys_str:
            raise ValueError("OPENAI_API_KEYS 为空！请在 .env 中配置多个 Key（逗号分隔）")

        self.keys = [k.strip() for k in keys_str.split(',') if k.strip()]
        self.base_url = base_url
        self.model = model
        self.clients: List[OpenAI] = []
        self.bad_keys = set()
        self.current_index = 0

        for key in self.keys:
            client = OpenAI(api_key=key, base_url=base_url)
            self.clients.append(client)

        logging.info(f"OpenAI客户端池初始化完成，共 {len(self.clients)} 个 Key（{len(self.keys)-len(self.bad_keys)} 个可用）")

    def get_next_client(self) -> Optional[OpenAI]:
        """轮询获取下一个可用客户端，跳过已标记为坏的"""
        if not self.clients:
            return None

        start_index = self.current_index
        attempts = 0
        while attempts < len(self.clients):
            client = self.clients[self.current_index]
            current_key = self.keys[self.current_index]

            if current_key not in self.bad_keys:
                self.current_index = (self.current_index + 1) % len(self.clients)
                return client

            # 这个 key 是坏的，尝试下一个
            self.current_index = (self.current_index + 1) % len(self.clients)
            attempts += 1

            if self.current_index == start_index:
                break  # 一圈都转完了，全是坏 key

        logging.error("所有 OpenAI API Key 均不可用！请检查余额或网络")
        return None

    def mark_bad(self, client: OpenAI):
        """标记当前客户端对应的 key 为坏的"""
        try:
            idx = self.clients.index(client)
            bad_key = self.keys[idx]
            if bad_key not in self.bad_keys:
                self.bad_keys.add(bad_key)
                logging.warning(f"API Key 已被标记为不可用（将跳过）：{bad_key[:10]}...{bad_key[-6:]}")
        except ValueError:
            pass

    def get_available_count(self) -> int:
        return len(self.keys) - len(self.bad_keys)
