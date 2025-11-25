from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import numpy as np
from deep_translator import GoogleTranslator

nltk.download('punkt')

class QuestionAnsweringSystem:
    def __init__(self, model_name="deepset/roberta-base-squad2", use_translation=True):
        self.use_translation = use_translation
        self.model_name = model_name
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = spacy.blank("en")

    def preprocess_context(self, context):
        sentences = sent_tokenize(context)
        return [s.strip() for s in sentences if s.strip()]

    def get_relevant_sentences(self, context, question, top_k=3):
        sentences = self.preprocess_context(context)
        if not sentences:
            return context
        q_doc = self.nlp(question)
        sent_docs = [self.nlp(s) for s in sentences]
        sims = []
        for sd in sent_docs:
            try:
                sims.append(q_doc.similarity(sd))
            except Exception:
                sims.append(0.0)
        top_k = min(top_k, len(sentences))
        top_indices = np.argsort(sims)[-top_k:]
        top_indices_sorted = sorted(top_indices)
        relevant = " ".join([sentences[i] for i in top_indices_sorted])
        return relevant if relevant else context

    def _translate_to_en(self, text):
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception:
            return text

    def _translate_from_en(self, text, target_lang):
        try:
            return GoogleTranslator(source='en', target=target_lang).translate(text)
        except Exception:
            return text

    def get_answer(self, context, question, top_k=3, target_lang=None):
        if self.use_translation and target_lang and target_lang.lower() != 'en':
            context_en = self._translate_to_en(context)
            question_en = self._translate_to_en(question)
        else:
            context_en = context
            question_en = question
        relevant_context = self.get_relevant_sentences(context_en, question_en, top_k=top_k)
        inputs = self.tokenizer(question_en, relevant_context, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits[0].cpu()
        end_logits = outputs.end_logits[0].cpu()
        start_idx = int(torch.argmax(start_logits).item())
        end_idx = int(torch.argmax(end_logits).item())
        if end_idx < start_idx:
            end_idx = start_idx
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        answer = self.tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx + 1]).strip()
        if self.use_translation and target_lang and target_lang.lower() != 'en' and answer:
            answer_translated = self._translate_from_en(answer, target_lang)
            return answer_translated
        return answer
