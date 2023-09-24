"""Discord bot to transcribe voice notes and reply with feedback"""
import os

import asynctempfile
import discord
import httpx
import openai
import tiktoken_async
import whisper


openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = whisper.load_model("base")


class MyClient(discord.Client):
    """Discord client"""

    async def on_ready(self):
        """Handle client ready event"""
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")

    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        if message.author == client.user:
            return

        channel = message.channel

        if "Direct Message" not in str(channel):
            await message.channel.send("I only reply in DMs")
            return

        if len(message.attachments) > 0:
            url = message.attachments[0].url

            async with httpx.AsyncClient() as httpx_client:
                resp = await httpx_client.get(url)
                b_content = resp.content

            async with asynctempfile.NamedTemporaryFile("wb+") as tmpfile:
                await tmpfile.write(b_content)
                await tmpfile.seek(0)

                result = await client.loop.run_in_executor(
                    None, MODEL.transcribe, tmpfile.name
                )
                transcript = result["text"]

            if transcript:
                # count tokens async
                encoder = await tiktoken_async.encoding_for_model("gpt-3.5-turbo")
                n_tokens = await client.loop.run_in_executor(
                    None, encoder.encode, transcript
                )
                if len(n_tokens) < 10:
                    return await message.channel.send(
                        "I can't understand that, please try again."
                    )

                msg = f"Hello! This is my speech. Let me know what I can improve.\n\n{transcript}"

                completion = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful teacher, judge, assistant. \
                                People come to you with speeches and explanations \
                                and you provide feedback about the content. From \
                                mistakes in the facts, to the structure of the speech. \
                                If possible you suggest other related topics to learn \
                                too. You always reply in the language of the speech.",
                        },
                        {"role": "user", "content": msg},
                    ],
                )
                respuesta = completion.choices[0].message["content"]
                await message.channel.send(respuesta.encode("utf-8").decode("utf-8"))
        else:
            await message.channel.send("I only reply to voice notes")


intents = discord.Intents.default()
intents.message_content = True  # pylint: disable=locally-disabled, multiple-statements, assigning-non-slot, line-too-long
client = MyClient(intents=intents)
client.run(os.getenv("DISCORD_TOKEN"))
