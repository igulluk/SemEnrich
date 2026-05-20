from .vqa_eval_acc import VQAEvalAcc 
import difflib 
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from collections import defaultdict
import nltk 
nltk.download('punkt')
import numpy as np 

class VQAEval:

    def __init__(self):
        self.vqa_acc = VQAEvalAcc()
        self.phase = None 
        self.gt_texts = []
        self.generated_texts = []
        self.mc_options = []

    def set_texts(self, step_outputs, phase):

        gt_texts = []
        generated_texts = []
        for x in step_outputs:
            gt_texts = gt_texts + x['gt_texts'] 
            generated_texts = generated_texts + x['generated_texts']         

        self.gt_texts = gt_texts
        self.generated_texts = generated_texts
        print(f'{len(gt_texts)=}')
        print(f'{len(generated_texts)=}')
        self.mc_options = []
        self.ans_type = []

        self.phase = phase
        
    def get_acc(self):
        acc_all = self.vqa_acc.evaluate(self.generated_texts, self.gt_texts)
        print(f'----- OPEN CASE -----')
        acc_open = self.vqa_acc.evaluate(
                [x for i, x in enumerate(self.generated_texts)],
                [x for i, x in enumerate(self.gt_texts)]
            )
        # print(f'----- CLOSED CASE -----')
        # acc_closed = self.vqa_acc.evaluate(
        #         [x for i, x in enumerate(self.generated_texts) if self.ans_type[i] == 'closed'],
        #         [x for i, x in enumerate(self.gt_texts) if self.ans_type[i] == 'closed']
        #     )
        
        acc_dict = {
            self.phase + f"_acc_all": acc_all,
            self.phase + f"_acc_open": acc_open,
            self.phase + f"_acc_closed": 0,
        }
        return acc_dict 
    

    def get_recall_open(self):
        generated_texts = [x for i, x in enumerate(self.generated_texts)]
        gt_texts = [x for i, x in enumerate(self.gt_texts)]

        ans = self.vqa_acc.evaluate_recall(generated_texts, gt_texts)
        
        dict_ = {
            self.phase + f"_recall_open": ans,
        }

        return dict_


    def get_bleu_scores(self):

        gt_texts = self.gt_texts
        generated_texts = self.generated_texts

        # Tokenize the texts
        def tokenize_text(text):
            """Tokenize text into words"""
            return nltk.word_tokenize(text.lower())

        gt_tokens = [tokenize_text(text) for text in gt_texts]
        generated_tokens = [tokenize_text(text) for text in generated_texts]


        def calculate_corpus_bleu(reference_corpus, candidate_corpus):
            """Calculate corpus-level BLEU score"""
            # Format references for corpus_bleu (each reference should be in a list)
            references = [[ref] for ref in reference_corpus]
            
            # Calculate BLEU scores for different n-grams
            bleu_1 = corpus_bleu(references, candidate_corpus, weights=(1, 0, 0, 0))
            bleu_2 = corpus_bleu(references, candidate_corpus, weights=(0.5, 0.5, 0, 0))
            bleu_3 = corpus_bleu(references, candidate_corpus, weights=(0.33, 0.33, 0.33, 0))
            bleu_4 = corpus_bleu(references, candidate_corpus, weights=(0.25, 0.25, 0.25, 0.25))
            
            return {
                'BLEU-1': bleu_1,
                'BLEU-2': bleu_2,
                'BLEU-3': bleu_3,
                'BLEU-4': bleu_4
            }

        corpus_scores = calculate_corpus_bleu(gt_tokens, generated_tokens)
        bleu_scores_dict = {self.phase + f"-CORPUS-{k}": v for k,v in corpus_scores.items()}

        # Calculate Sentence-level BLEU Scores
        def calculate_sentence_bleu(reference_corpus, candidate_corpus):
            """Calculate sentence-level BLEU scores"""
            smoothing = SmoothingFunction().method1
            
            sentence_scores = []
            
            for ref, cand in zip(reference_corpus, candidate_corpus):
                # Calculate BLEU scores for different n-grams
                bleu_1 = sentence_bleu([ref], cand, weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu_2 = sentence_bleu([ref], cand, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu_3 = sentence_bleu([ref], cand, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                bleu_4 = sentence_bleu([ref], cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                
                sentence_scores.append({
                    'BLEU-1': bleu_1,
                    'BLEU-2': bleu_2,
                    'BLEU-3': bleu_3,
                    'BLEU-4': bleu_4
                })
            

            return {
                'BLEU-1': np.array([x['BLEU-1'] for x in sentence_scores]).mean(),
                'BLEU-2': np.array([x['BLEU-2'] for x in sentence_scores]).mean(),
                'BLEU-3': np.array([x['BLEU-3'] for x in sentence_scores]).mean(),
                'BLEU-4': np.array([x['BLEU-4'] for x in sentence_scores]).mean(),
            }
            
        sentence_scores = calculate_sentence_bleu(gt_tokens, generated_tokens)
        bleu_scores_dict.update({self.phase + f"-SENT-{k}": v for k,v in sentence_scores.items()})

        return bleu_scores_dict


    # def get_bleu_scores(self):

    #     gt_texts = [[x] for x in self.gt_texts[:5]]
    #     generated_texts = self.generated_texts[:5]
    #     print(f'gt_texts : {gt_texts}')
    #     print(f'generated_texts : {generated_texts}')
    #     print(f'gt_texts : {type(gt_texts)}')
    #     print(f'generated_texts : {type(generated_texts)}')
    #     bleu_score1 = corpus_bleu(gt_texts, generated_texts, weights=(1, 0, 0, 0))
    #     bleu_score2 = corpus_bleu(gt_texts, generated_texts, weights=(0.5, 0.5, 0, 0))
    #     bleu_score3 = corpus_bleu(gt_texts, generated_texts, weights=(1/3, 1/3, 1/3, 0))
    #     bleu_score4 = corpus_bleu(gt_texts, generated_texts, weights=(0.25, 0.25, 0.25, 0.25))
    #     bleu_scores = [bleu_score1, bleu_score2, bleu_score3, bleu_score4]
    #     bleu_scores_dict = {
    #         self.phase + f"_corpusbleu_{i+1}": bleu_scores[i] for i in range(4)
    #     }

    #     cc = SmoothingFunction()

    #     sentence_blue_avg1 = 0
    #     sentence_blue_avg2 = 0
    #     sentence_blue_avg3 = 0
    #     sentence_blue_avg4 = 0
    #     for i in range(len(gt_texts)):
    #         x = gt_texts[i]
    #         y = generated_texts[i]
    #         print(f'x : {x}')
    #         print(f'y : {y}')
    #         print(f'x : {type(x)}')
    #         print(f'y : {type(y)}')
    #         if x[0] == '' and len(x)==1:
    #             continue

    #         assert False 
    #         sentence_blue_avg1 += sentence_bleu(x, y, weights=(1, 0, 0, 0), smoothing_function=cc.method4) 
    #         sentence_blue_avg2 += sentence_bleu(x, y, weights=(0.5, 0.5, 0, 0), smoothing_function=cc.method4) 
    #         sentence_blue_avg3 += sentence_bleu(x, y, weights=(1/3, 1/3, 1/3, 0), smoothing_function=cc.method4) 
    #         sentence_blue_avg4 += sentence_bleu(x, y, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method4) 
    #     # for i in range(len(gt_texts)):
    #     #     x = gt_texts[i][0]
    #     #     y = generated_texts[i]
    #     #     if x == '':
    #     #         continue
    #     #     sentence_blue_avg1 += sentence_bleu(x, y, weights=(1, 0, 0, 0), smoothing_function=cc.method4) 
    #     #     sentence_blue_avg2 += sentence_bleu(x, y, weights=(0.5, 0.5, 0, 0), smoothing_function=cc.method4) 
    #     #     sentence_blue_avg3 += sentence_bleu(x, y, weights=(1/3, 1/3, 1/3, 0), smoothing_function=cc.method4) 
    #     #     sentence_blue_avg4 += sentence_bleu(x, y, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method4) 

    #     bleu_scores = [sentence_blue_avg1, sentence_blue_avg2, sentence_blue_avg3, sentence_blue_avg4]
    #     bleu_scores = [x/len(gt_texts) for x in bleu_scores]
    #     bleu_scores_dict.update({
    #         self.phase + f"_sentavgbleu_{i+1}": bleu_scores[i] for i in range(4)
    #     })

    #     return bleu_scores_dict
        

    def get_cider_score(self):

        gt_texts = [[x] for x in self.gt_texts]
        generated_texts = [[x] for x in self.generated_texts]
        
        gt_texts = {i:gt_texts[i] for i in range(len(gt_texts))}
        generated_texts = {i:generated_texts[i] for i in range(len(generated_texts))}

        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gt_texts, generated_texts) 
        cider_score_dict = {
            self.phase + f"_cider_score": cider_score
        }

        return cider_score_dict


    def get_rouge_scores(self):

        gt_texts = self.gt_texts
        generated_texts = self.generated_texts
        assert isinstance(gt_texts[0], str)
        assert isinstance(generated_texts[0], str)
        n = len(generated_texts)

        rouges = ['rouge1', 'rouge2', 'rougeL']
        metrics = ['precision', 'recall', 'fmeasure']
        scores_dict = defaultdict(int)

        scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
        for i in range(len(gt_texts)):
            scores = scorer.score(gt_texts[i], generated_texts[i])
            for r in rouges:
                for m in metrics:
                    scores_dict['_'+r+'_'+m] += getattr(scores[r], m)/n

        scores_dict = {self.phase+k:v for k,v in scores_dict.items()}

        return scores_dict
        

    def get_meteor_score(self):

        gt_texts = self.gt_texts
        generated_texts = self.generated_texts
        n = len(gt_texts)

        meteor_score_ = 0
        for hypothesis, reference_list in zip(generated_texts, gt_texts):

            reference_list = reference_list.split(' ')
            hypothesis = hypothesis.split(' ')
            reference_list = [s for s in reference_list if s.strip()]
            hypothesis = [s for s in hypothesis if s.strip()]
            reference_list = [reference_list]
            meteor_score_ += meteor_score(reference_list, hypothesis)/n

        meteor_score_dict = {
            self.phase + f"_meteor_score": meteor_score_
        }

        return meteor_score_dict

    def get_mc_accuracy(self):

        if self.mc_options[0][0] is None :
            return {}
        gt_texts = self.gt_texts
        generated_texts = self.generated_texts 
        mc_options = self.mc_options

        n = len(gt_texts)
        correct = 0 
        for i in range(n):
            index_pred = self.find_most_similar_index(mc_options[i],generated_texts[i])
            index_label  = self.find_most_similar_index(mc_options[i],gt_texts[i])
            if (index_pred is not None) and (index_label is not None) and index_pred == index_label:
                correct += 1
            if (index_pred is None) or (index_label is None):
                print(f'index_pred : {index_pred}')
                print(f'index_label : {index_label}')
                print(f'mc_options[i] : {mc_options[i]}')
                print(f'gt_texts[i] : {gt_texts[i]}')
                print(f'generated_texts[i] : {generated_texts[i]}')
        
        mc_acc_dict = {
            self.phase + f"_mc_acc": correct/n
        }
        return mc_acc_dict


    def str_similarity(self, str1, str2):
        seq = difflib.SequenceMatcher(None, str1, str2)
        return seq.ratio()
    
    def find_most_similar_index(self, str_list, target_str):
        """
        Given a list of strings and a target string, returns the index of the most similar string in the list.
        """
        # Initialize variables to keep track of the most similar string and its index
        most_similar_str = None
        most_similar_index = None
        highest_similarity = 0
        
        # Iterate through each string in the list
        for i, str_ in enumerate(str_list):
            # Calculate the similarity between the current string and the target string
            similarity = self.str_similarity(str_, target_str)
            
            # If the current string is more similar than the previous most similar string, update the variables
            if similarity > highest_similarity:
                most_similar_str = str_
                most_similar_index = i
                highest_similarity = similarity
        
        # Return the index of the most similar string
        return most_similar_index


    def get_results(self, step_outputs, phase):
        self.set_texts(step_outputs, phase)
        res = {}
        res.update(self.get_bleu_scores())
        res.update(self.get_cider_score())
        res.update(self.get_rouge_scores())
        res.update(self.get_meteor_score())
        # res.update(self.get_mc_accuracy())
        res.update(self.get_acc())
        res.update(self.get_recall_open())
        return res 