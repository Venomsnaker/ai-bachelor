from openai import OpenAI
from groq import Groq
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import os
from dotenv import load_dotenv

class SelfCheckAPIPrompt:
    def __init__(
        self,
        client_type = "openai",
        model = "gpt-3.5-turbo",
        api_key = None,
    ):
        if client_type == "openai":
            self.client = OpenAI()
            print("Initiate OpenAI client... model = {}".format(model)) 
        elif client_type == "groq":
            self.client = Groq(api_key=api_key)
            print("Initiate Groq client ... model = {}".format(model))

        self.client_type = client_type
        self.model = model
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    def completion(self, prompt: str):
        if self.client_type == "openai" or self.client_type == "groq":
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=5
            )
            return chat_completion['choices'][0]['message']['content']
        else:
            raise ValueError("client_type not implemented")
        
    def predict(
        self, 
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose

        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]

            for sample_i, sample in enumerate(sampled_passages):
                sample = sample.replace("\n", " ") 
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                generate_text = self.completion(prompt)
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
            self,
            text,
    ):
        text = text.lower().strip()
        
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]
