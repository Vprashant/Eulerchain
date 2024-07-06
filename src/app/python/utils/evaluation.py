from dataclasses import dataclass
from functools import reduce
from typing import List, Tuple
from src.app.python.constant.project_constant import Constant as constant
from src.app.python.constant.global_data import GlobalData
import torch
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from nltk.util import ngrams

@dataclass
class EvaluationResult:
    BLEU: float
    ROUGE1: float
    BERTP: float
    BERTR: float
    BERTF1: float
    Perplexity: float
    Diversity: float
    classified_result: dict

def evaluator_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        bert_scores = [val for key, val in result.items() if "BERT" in key]
        similarity_score = result["BLEU"] * 0.2 + result["ROUGE1"] * 0.3 + (reduce(lambda a, b: a + b, bert_scores) / len(bert_scores)) * 0.5
        uniqueness_score = result["Perplexity"] * 0.6 + result["Diversity"] * 0.4
        result["classified_result"] = {"similarity_score": similarity_score, "uniqueness_score": uniqueness_score}
        return EvaluationResult(**result)
    return wrapper

class RAGEvaluator:
    def __init__(self):
        self.gpt2_model, self.gpt2_tokenizer = self.load_gpt2_model()
        self.bias_pipeline = pipeline("zero-shot-classification", model="Hate-speech-CNERG/dehatebert-mono-english")

    def load_gpt2_model(self):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return model, tokenizer

    def evaluate_bleu_rouge(self, candidates: List[str], references: List[str]) -> Tuple[float, float]:
        bleu_score = corpus_bleu(candidates, [references]).score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        return bleu_score, rouge1

    def evaluate_bert_score(self, candidates: List[str], references: List[str]) -> Tuple[float, float, float]:
        P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
        return P.mean().item(), R.mean().item(), F1.mean().item()

    def evaluate_perplexity(self, text: str) -> float:
        encodings = self.gpt2_tokenizer(text, return_tensors='pt')
        max_length = self.gpt2_model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    def evaluate_diversity(self, texts: List[str]) -> float:
        all_tokens = [tok for text in texts for tok in text.split()]
        unique_bigrams = set(ngrams(all_tokens, 2))
        diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
        return diversity_score

    def evaluate_bias(self, response, question):
        text = f"{response} {question}"
        result = self.bias_pipeline (text)
        label = result[0]['label']
        score = result[0]['score']
        return label, score

    @evaluator_decorator
    def evaluate_all(self, question: str, response: str, reference: str) -> EvaluationResult:
        candidates = [response]
        references = [reference]
        bleu, rouge1 = self.evaluate_bleu_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        perplexity = self.evaluate_perplexity(response)
        diversity = self.evaluate_diversity(candidates)
        # racial_bias = self.evaluate_racial_bias(response, question)
        result = {
            "BLEU": bleu,
            "ROUGE1": rouge1,
            "BERTP": bert_p,
            "BERTR": bert_r,
            "BERTF1": bert_f1,
            "Perplexity": perplexity,
            "Diversity": diversity,
        }
        return result

rag_eval = RAGEvaluator()
