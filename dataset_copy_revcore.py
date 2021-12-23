import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
from collections import defaultdict
from random import shuffle
import random


def _edge_list_1(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if (
                    entity != tail_and_relation[1] and tail_and_relation[0] != 185
                ):  # and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append(
                        (entity, tail_and_relation[1], tail_and_relation[0])
                    )
                    edge_list.append(
                        (tail_and_relation[1], entity, tail_and_relation[0])
                    )

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [
        (h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000
    ], relation_cnt


def get_neighborhood(edge_list, entity_id, max_neighbors=30):
    neighbors = [triplet[1] for triplet in edge_list if triplet[0] == entity_id]
    return neighbors[:max_neighbors]


def get_2_hops_neighbors_via_kg(
    kg,
    entity_id,
    relation_counts,
    max_neighbors=5,
    type_sampling="random",
    node_degree=None,
):
    #     one_hops_neighbors = [t[1] for t in kg[entity_id] if t[1] != entity_id and relation_counts[int(t[0])] > 1000]
    one_hops_neighbors = [t[1] for t in kg[entity_id] if t[1] != entity_id]
    two_hops_neighbors = [
        t[1]
        for n_id in one_hops_neighbors
        for t in kg[n_id]
        if t[1] != entity_id and relation_counts[t[0]] > 1000
    ]

    if type_sampling == "random":
        random.shuffle(one_hops_neighbors)
    elif type_sampling == "relation":
        one_hops_neighbors = [
            t[1]
            for t in kg[entity_id]
            if t[1] != entity_id and relation_counts[int(t[0])] > 1000
        ]
    elif type_sampling == "degree":
        one_hops_neighbors = [(x, node_degree[x]) if x in node_degree else (x, 0) for x in one_hops_neighbors]
        sorted(one_hops_neighbors, key=lambda x: x[1], reverse=True)
        one_hops_neighbors = [x[0] for x in one_hops_neighbors]
    return one_hops_neighbors[:max_neighbors], two_hops_neighbors[:max_neighbors]


class dataset(object):
    def __init__(self, filename, opt):
        self.entity2entityId = pkl.load(open("data/entity2entityId.pkl", "rb"))
        self.entity_max = len(self.entity2entityId)

        self.id2entity = pkl.load(open("data/id2entity.pkl", "rb"))
        self.subkg = pkl.load(open("data/subkg.pkl", "rb"))  # need not back process
        self.text_dict = pkl.load(open("data/text_dict.pkl", "rb"))

        self.batch_size = opt["batch_size"]
        self.max_c_length = opt["max_c_length"]
        self.max_r_length = opt["max_r_length"]
        self.max_count = opt["max_count"]
        self.entity_num = opt["n_entity"]
        self.max_neighbors = opt["max_neighbors"]
        # self.word2index=json.load(open('word2index.json',encoding='utf-8'))

        f = open(filename, encoding="utf-8")
        self.data = []
        self.corpus = []
        for line in tqdm(f):
            lines = json.loads(line.strip())
            seekerid = lines["initiatorWorkerId"]
            recommenderid = lines["respondentWorkerId"]
            contexts = lines["messages"]
            movies = lines["movieMentions"]
            altitude = lines["respondentQuestions"]
            initial_altitude = lines["initiatorQuestions"]
            cases = self._context_reformulate(
                contexts, movies, altitude, initial_altitude, seekerid, recommenderid
            )
            self.data.extend(cases)

        # if 'train' in filename:

        # self.prepare_word2vec()
        self.word2index = json.load(open("word2index_redial.json", encoding="utf-8"))
        self.key2index = json.load(open("key2index_3rd.json", encoding="utf-8"))

        self.stopwords = set(
            [word.strip() for word in open("stopwords.txt", encoding="utf-8")]
        )
        
        self.review2entities = json.load(open('data/dbpedia_review_entities.json'))

        self.edge_list, self.relation_counts = _edge_list_1(self.subkg, 64368, hop=2)

        self.node_degree = {k: len(v) for k, v in self.subkg.items()}

        self.type_sampling = opt["type_sampling"]

        if self.max_neighbors > 0:
            print(
                f"use neighborhood incoporation ..... num neighbors: {self.max_neighbors}"
            )

        # self.co_occurance_ext(self.data)
        # exit()

    def prepare_word2vec(self):
        import gensim

        model = gensim.models.word2vec.Word2Vec(self.corpus, size=300, min_count=1)
        model.save("word2vec_redial")
        word2index = {word: i + 4 for i, word in enumerate(model.wv.index2word)}
        # word2index['_split_']=len(word2index)+4
        # json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
        word2embedding = (
            [[0] * 300] * 4 + [model[word] for word in word2index] + [[0] * 300]
        )
        import numpy as np

        word2index["_split_"] = len(word2index) + 4
        json.dump(
            word2index,
            open("word2index_redial.json", "w", encoding="utf-8"),
            ensure_ascii=False,
        )

        print(np.shape(word2embedding))
        np.save("word2vec_redial.npy", word2embedding)

    def padding_w2v(self, sentence, max_length, transformer=True, pad=0, end=2, unk=3):
        vector = []
        concept_mask = []
        dbpedia_mask = []
        reviews_mask = []

        for word in sentence:
            vector.append(self.word2index.get(word, unk))
            # if word.lower() not in self.stopwords:
            concept_mask.append(self.key2index.get(word.lower(), 0))
            # else:
            #    concept_mask.append(0)
            if "@" in word:
                try:
                    entity = self.id2entity[int(word[1:])]
                    id = self.entity2entityId[entity]
                except:
                    id = self.entity_max
                dbpedia_mask.append(id)
            else:
                dbpedia_mask.append(self.entity_max)

            #### review_mask ####
            if '#' in word:
                reviews_mask.append(1)#self.word2index.get(word[1:],unk))
            else:
                reviews_mask.append(0)#pad)

        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)
        reviews_mask.append(0)#pad)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:],reviews_mask[:max_length]
            else:
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length],reviews_mask[:max_length]
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],\
                   dbpedia_mask+(max_length-len(vector))*[self.entity_max],\
                   reviews_mask+(max_length-len(vector))*[0]

    def padding_context(self, contexts, pad=0, transformer=True):
        vectors = []
        vec_lengths = []
        if transformer == False:
            if len(contexts) > self.max_count:
                for sen in contexts[-self.max_count :]:
                    vec, v_l = self.padding_w2v(sen, self.max_r_length, transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors, vec_lengths, self.max_count
            else:
                length = len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen, self.max_r_length, transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return (
                    vectors + (self.max_count - length) * [[pad] * self.max_c_length],
                    vec_lengths + [0] * (self.max_count - length),
                    length,
                )
        else:
            contexts_com = []
            for sen in contexts[-self.max_count : -1]:
                contexts_com.extend(sen)
                contexts_com.append("_split_")
            contexts_com.extend(contexts[-1])
            vec, v_l, concept_mask, dbpedia_mask, reviews_mask = self.padding_w2v(
                contexts_com, self.max_c_length, transformer
            )
            return vec, v_l, concept_mask, dbpedia_mask, reviews_mask, 0

    def response_delibration(self, response, unk="MASKED_WORD"):
        new_response = []
        for word in response:
            if word in self.key2index:
                new_response.append(unk)
            else:
                new_response.append(word)
        return new_response

    def data_process(self, is_finetune=False):
        data_set = []
        context_before = []
        for line in self.data:
            # if len(line['contexts'])>2:
            #    continue
            if is_finetune and line["contexts"] == context_before:
                continue
            else:
                context_before = line["contexts"]
            context, c_lengths, concept_mask, dbpedia_mask, reviews_mask,  _ = self.padding_context(
                line["contexts"]
            )
            response, r_length, _, _ = self.padding_w2v(
                line["response"], self.max_r_length
            )
            if False:
                mask_response, mask_r_length, _, _ = self.padding_w2v(
                    self.response_delibration(line["response"]), self.max_r_length
                )
            else:
                mask_response, mask_r_length = response, r_length
            assert len(context) == self.max_c_length
            assert len(concept_mask) == self.max_c_length
            assert len(dbpedia_mask) == self.max_c_length
            
            # review-entities intercoporation
            new_entities = []
            for entity in line['entity']:
                if str(entity) in self.review2entities:
                    review_entities = self.review2entities[str(entity)]
                    review_entities = [int(x) for x in review_entities]
                    new_entities.extend(review_entities)

            data_set.append(
                [
                    np.array(context),
                    c_lengths,
                    np.array(response),
                    r_length,
                    np.array(mask_response),
                    mask_r_length,
                    line["entity"] + new_entities,
                    line["movie"],
                    concept_mask,
                    dbpedia_mask,
                    reviews_mask,
                    line["rec"],
                ]
            )
        return data_set

    def co_occurance_ext(self, data):
        stopwords = set(
            [word.strip() for word in open("stopwords.txt", encoding="utf-8")]
        )
        keyword_sets = set(self.key2index.keys()) - stopwords
        movie_wordset = set()
        for line in data:
            movie_words = []
            if line["rec"] == 1:
                for word in line["response"]:
                    if "@" in word:
                        try:
                            num = self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            line["movie_words"] = movie_words
        new_edges = set()
        for line in data:
            if len(line["movie_words"]) > 0:
                before_set = set()
                after_set = set()
                co_set = set()
                for sen in line["contexts"]:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in line["response"]:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in line["movie_words"]:
                    for word in list(before_set):
                        new_edges.add("co_before" + "\t" + movie + "\t" + word + "\n")
                    for word in list(co_set):
                        new_edges.add(
                            "co_occurance" + "\t" + movie + "\t" + word + "\n"
                        )
                    for word in line["movie_words"]:
                        if word != movie:
                            new_edges.add(
                                "co_occurance" + "\t" + movie + "\t" + word + "\n"
                            )
                    for word in list(after_set):
                        new_edges.add("co_after" + "\t" + word + "\t" + movie + "\n")
                        for word_a in list(co_set):
                            new_edges.add(
                                "co_after" + "\t" + word + "\t" + word_a + "\n"
                            )
        f = open("co_occurance.txt", "w", encoding="utf-8")
        f.writelines(list(new_edges))
        f.close()
        json.dump(
            list(movie_wordset),
            open("movie_word.json", "w", encoding="utf-8"),
            ensure_ascii=False,
        )
        print(len(new_edges))
        print(len(movie_wordset))

    def entities2ids(self, entities):
        return [self.entity2entityId[word] for word in entities]

    def detect_movie(self, sentence, movies):
        token_text = word_tokenize(sentence)
        num = 0
        token_text_com = []
        while num < len(token_text):
            if token_text[num] == "@" and num + 1 < len(token_text):
                token_text_com.append(token_text[num] + token_text[num + 1])
                num += 2
            else:
                token_text_com.append(token_text[num])
                num += 1
        movie_rec = []
        for word in token_text_com:
            if word[1:] in movies:
                movie_rec.append(word[1:])
        movie_rec_trans = []
        for movie in movie_rec:
            entity = self.id2entity[int(movie)]
            try:
                movie_rec_trans.append(self.entity2entityId[entity])
            except:
                pass
        return token_text_com, movie_rec_trans

    def _context_reformulate(
        self, context, movies, altitude, ini_altitude, s_id, re_id
    ):
        last_id = None
        # perserve the list of dialogue
        context_list = []
        for message in context:
            entities = []
            try:
                for entity in self.text_dict[message["text"]]:
                    try:
                        entities.append(self.entity2entityId[entity])
                    except:
                        pass
            except:
                pass
            token_text, movie_rec = self.detect_movie(message["text"], movies)
            if len(context_list) == 0:
                context_dict = {
                    "text": token_text,
                    "entity": entities + movie_rec,
                    "user": message["senderWorkerId"],
                    "movie": movie_rec,
                }
                context_list.append(context_dict)
                last_id = message["senderWorkerId"]
                continue
            if message["senderWorkerId"] == last_id:
                context_list[-1]["text"] += token_text
                context_list[-1]["entity"] += entities + movie_rec
                context_list[-1]["movie"] += movie_rec
            else:
                context_dict = {
                    "text": token_text,
                    "entity": entities + movie_rec,
                    "user": message["senderWorkerId"],
                    "movie": movie_rec,
                }
                context_list.append(context_dict)
                last_id = message["senderWorkerId"]

        cases = []
        contexts = []
        entities_set = set()
        entities = []
        for context_dict in context_list:
            self.corpus.append(context_dict["text"])
            if context_dict["user"] == re_id and len(contexts) > 0:
                response = context_dict["text"]

                # entity_vec=np.zeros(self.entity_num)
                # for en in list(entities):
                #    entity_vec[en]=1
                # movie_vec=np.zeros(self.entity_num+1,dtype=np.float)
                if len(context_dict["movie"]) != 0:
                    for movie in context_dict["movie"]:
                        # if movie not in entities_set:
                        cases.append(
                            {
                                "contexts": deepcopy(contexts),
                                "response": response,
                                "entity": deepcopy(entities),
                                "movie": movie,
                                "rec": 1,
                            }
                        )
                else:
                    cases.append(
                        {
                            "contexts": deepcopy(contexts),
                            "response": response,
                            "entity": deepcopy(entities),
                            "movie": 0,
                            "rec": 0,
                        }
                    )

                contexts.append(context_dict["text"])
                for word in context_dict["entity"]:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
            else:
                contexts.append(context_dict["text"])
                for word in context_dict["entity"]:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
        return cases


class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num):
        self.data = dataset
        self.entity_num = entity_num
        self.concept_num = concept_num + 1

    def __getitem__(self, index):
        """
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        """
        (
            context,
            c_lengths,
            response,
            r_length,
            mask_response,
            mask_r_length,
            entity,
            movie,
            concept_mask,
            dbpedia_mask,
            reviews_mask,
            rec,
        ) = self.data[index]
        entity_vec = np.zeros(self.entity_num)
        entity_vector = np.zeros(200, dtype=np.int)
        point = 0
        for en in entity:
            entity_vec[en] = 1
            entity_vector[point] = en
            point += 1

        concept_vec = np.zeros(self.concept_num)
        for con in concept_mask:
            if con != 0:
                concept_vec[con] = 1

        db_vec = np.zeros(self.entity_num)
        for db in dbpedia_mask:
            if db != 0:
                db_vec[db] = 1
        # for db in dbpedia_mask:
        #     if db!=0:
        #         db_vec[db]=1

        return (
            context,
            c_lengths,
            response,
            r_length,
            mask_response,
            mask_r_length,
            entity_vec,
            entity_vector,
            movie,
            np.array(concept_mask),
            np.array(dbpedia_mask),
            np.array(reviews_mask),
            concept_vec,
            db_vec,
            rec,
        )

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    ds = dataset("data/train_data.jsonl")
    print()