import spacy
import bert_score
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
from transformers import logging
logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LongformerTokenizer, LongformerForMultipleChoice, LongformerForSequenceClassification
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from selfcheckgpt.utils import MQAGConfig, expand_list1, expand_list2, NLIConfig, LLMPromptConfig
from modeling_mqag import question_generation_sentence_level, answering
from modeling_ngram import UnigramModel, NgramModel
from modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

def method_simple_counting(
        prob,
        u_score,
        prob_s,
        u_score_s,
        num_samples,
        AT,
):
    """
    Simple Count Method: count_mismatch / (count_match + count_mismatch)
    """
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_good_sample, count_match = 0, 0

    for s in range(num_samples):
        if u_score_s[s] >= AT:
            count_good_sample += 1
            a_S = np.argmax(prob_s[s])

            if a_DT == a_S:
                count_match += 1
    if count_good_sample == 0:
        score = 0.5
    else:
        score = (count_good_sample - count_match) / count_good_sample
    return score

def method_vanilla_bayes(
        prob,
        u_score,
        prob_s,
        u_score_s,
        num_samples,
        beta1,
        beta2,
        AT,
):
    """
    Vanilla Bayes Method: Compute P(sentence is non-factual | count_match, count_mismatch)
    """
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0

    for s in range(num_samples):
        if u_score_s[s] >= AT:
            a_S = np.argmax(prob_s[s])

            if a_DT == a_S:
                count_match += 1
            else:
                count_mismatch += 1
    gamma1 = beta2 / (1.0 - beta1)
    gamma2 = beta1 / (1.0 - beta2)
    score = (gamma2 ** count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def method_bayes_with_alpha(
        prob,
        u_score,
        prob_s,
        u_score_s,
        num_samples,
        beta1, beta2,
):
    """
    Bayes Method: Soft-counting
    """
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0

    for s in range(num_samples):
        ans_score = u_score_s[s]
        a_S = np.argmax(prob_s[s])

        if a_DT == a_S:
            count_match += ans_score
        else:
            count_mismatch += ans_score
    gamma1 = beta2 / (1.0 - beta1)
    gamma2 = beta1 / (1.0 - beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def answerability_scoring(
        u_model,
        u_tokenizer,
        question,
        context,
        max_length,
        device,
):
    """
    Prob -> 0.0: unanswerable, Prob -> 1.0: answerable
    """
    input_text = question + ' ' + u_tokenizer.sep_token + ' ' + context
    inputs = u_tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    logits = u_model(**inputs).logits
    logits = logits.squeeze(-1)
    prob = torch.sigmoid(logits).item()
    return prob

class SelfCheckMQAG:
    def __init__(
            self,
            g1_model: str = None,
            g2_model: str = None,
            answering_model: str = None,
            answerability_model: str = None,
            device = None
    ):
        g1_model = g1_model if g1_model is not None else MQAGConfig.generation1_squad
        g2_model = g2_model if g2_model is not None else MQAGConfig.generation2
        answering_model = answering_model if answering_model is not None else MQAGConfig.answering
        answerability_model = answerability_model if answerability_model is not None else MQAGConfig.answerability

        self.g1_tokenizer = AutoTokenizer.from_pretrained(g1_model)
        self.g1_model = AutoModelForSeq2SeqLM.from_pretrained(g1_model)
        self.g2_tokenizer = AutoTokenizer.from_pretrained(g2_model)
        self.g2_model = AutoModelForSeq2SeqLM.from_pretrained(g2_model)
        self.a_tokenizer = LongformerTokenizer.from_pretrained(answering_model)
        self.a_model = LongformerForMultipleChoice.from_pretrained(answering_model)
        self.u_tokenizer = LongformerTokenizer.from_pretrained(answerability_model)
        self.u_model = LongformerForSequenceClassification.from_pretrained(answerability_model)

        self.g1_model.eval()
        self.g2_model.eval()
        self.a_model.eval()
        self.u_model.eval()

        if device is None:
            device = torch.device("cpu")
        self.g1_model.to(device)
        self.g2_model.to(device)
        self.a_model.to(device)
        self.u_model.to(device)
        self.device = device
        print("SelfCheck-MQAG initialized to device", device)
    
    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        passage: str,
        sampled_passages: List[str],
        num_questions_per_sent: int = 5,
        scoring_method: str = "bayes_with_alpha", **kwargs,
        ):
            assert scoring_method in ['counting', 'bayes', 'bayes_with_alpha']
            num_samples = len(sampled_passages)
            sent_scores = []
            for sentence in sentences:
                questions = question_generation_sentence_level(
                    self.g1_model, self.g1_tokenizer,
                    self.g2_model, self.g2_tokenizer,
                    sentence, passage, num_questions_per_sent, self.device)
                scores = []
                max_seq_length = 4096

                for question_item in questions:
                    question, options = question_item['question'], question_item['options']
                    prob = answering(
                        self.a_model, self.a_tokenizer,
                        question, options, passage,
                        max_seq_length, self.device)
                    u_score = answerability_scoring(
                        self.u_model, self.u_tokenizer,
                        question, passage,
                        max_seq_length, self.device)
                    prob_s = np.zeros((num_samples, 4))
                    u_score_s = np.zeros((num_samples, ))

                    for si, sampled_passage in enumerate(sampled_passages):
                        prob_s[si] = answering(
                            self.a_model, self.a_tokenizer,
                            question, options, sampled_passage,
                            max_seq_length, self.device)
                        u_score_s[si] = answerability_scoring(
                            self.u_model, self.u_tokenizer,
                            question, sampled_passage,
                            max_seq_length, self.device)
                    if scoring_method == 'counting':
                        score = method_simple_counting(prob, u_score, prob_s, u_score_s, num_samples, AT=kwargs['AT'])
                    elif scoring_method == 'bayes':
                        score = method_vanilla_bayes(prob, u_score, prob_s, u_score_s, num_samples, beta1=kwargs['beta1'], beta2=kwargs['beta2'], AT=kwargs['AT'])
                    elif scoring_method == 'bayes_with_alpha':
                        score = method_bayes_with_alpha(prob, u_score, prob_s, u_score_s, num_samples, beta1=kwargs['beta1'], beta2=kwargs['beta2'])
                    scores.append(score)
                sent_score = np.mean(scores)
                sent_scores.append(sent_score)
            return np.array(sent_scores)

class SelfCheckBERTScore:
    def __init__(self, default_model="en", rescale_with_baseline=True):
        self.nlp = spacy.load("en_core_web_sm")
        self.default_model = default_model
        self.rescale_with_baseline = rescale_with_baseline
        print("SelfCheck-BERTScore initializedd")

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))

        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(sample_passage).sents]
            sentences_sample = [sent.text.strip() for sent in sentences_sample if len(sent) > 3]
            num_sentences_sample = len(sentences_sample)
            refs = expand_list1(sentences, num_sentences_sample)
            cands = expand_list2(sentences_sample, num_sentences)

            P, R, F1 = bert_score.score(
                cands, refs,
                lang=self.default_model,
                verbose=False,
                rescale_with_baseline=self.rescale_with_baseline
            )
            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()
            bertscore_array[:,s] = F1_arr_max_axis1
        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent

class SelfCheckNgram:
    def __init__(self, n: int, lowercase: bool = True):
        self.n = n
        self.lowercase = lowercase
        print(f"SelfCheck-{n}gram initialized")

    def predict(
            self, 
            sentences: List[str],
            passage: str,
            sampled_passages: List[str],
    ):
        if self.n == 1:
            ngram_model = UnigramModel(lowercase=self.lowercase)
        elif self.n > 1:
            ngram_model = NgramModel(n=self.n, lowercase=self.lowercase)
        else:
            raise ValueError("n must be integer >= 1")
        ngram_model.add(passage)
        
        for sampled_passage in sampled_passages:
            ngram_model.add(sampled_passage)
        ngram_model.train(k=0)
        ngram_pred = ngram_model.evaluate(sentences)
        return ngram_pred
    
class SelfCheckNLI:
    def __init__(
            self,
            nli_model: str = None,
            device = None
    ):
        nli_model = nli_model if nli_model is not None else NLIConfig.nli_model
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(nli_model)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        print("SelfCheck-NLI initialized to device", device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))

        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(sentence, sample)],
                    add_special_tokens=True, padding="longest",
                    truncation=True, return_tensors="pt",
                    return_token_type_ids=True, return_attention_mask=True,
                )
                inputs = inputs.to(self.device)
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                prob_ = probs[0][1].item()
                scores[sent_i, sample_i] = prob_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence
      
class SelfCheckLLMPrompt:
    def __init__(
            self,
            model: str = None,
            device = None
    ):
        model = model if model is not None else ""
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
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
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False
                )
                ouput_text = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = ouput_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(aixs=-1)
        return scores_per_sentence

    def text_postprocessing(
            self,
            text,
    ):
        text = text.lower().strip()

        if text[:3] == "yes":
            text = 'yes'
        elif text[:2] == "no":
            text = "no"
        else:
            if text not in self.not_defined_text:
                print(f"Warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]