import yaml
import re
from melon_clustering import SENTENCES_DIR
import stanza
from pathlib import Path
from collections import defaultdict, OrderedDict

_nlp_stanza = None

class Loader:
    @staticmethod
    def load_db_sentences(lemma, language):
        from melon_db import Parser
        parser = Parser(language)
        res = parser.get_query(
            """
            WITH RankedSegments AS (
                SELECT l.lemma, ts.text, w.word, sn.name AS source_name, se.name AS source_entry_name, ts.index,
                        ROW_NUMBER() OVER (PARTITION BY l.lemma, sn.name, sn.domain_subcategory_id ORDER BY se.index, ts.index) AS rn
                FROM text_segments ts
                JOIN source_entries se ON ts.source_entry_id = se.id
                JOIN words_in_text_segments wits ON ts.id = wits.text_segment_id
                JOIN words w ON wits.word_id = w.id
                JOIN lemmas l ON w.lemma_id = l.id
                JOIN source_names sn ON se.source_name_id = sn.id
                WHERE l.lemma = %s AND sn.lang = %s)
            SELECT word, text, source_entry_name
            FROM RankedSegments
            ORDER BY lemma, source_entry_name, index;""", (lemma, language)
        )
        result = {}
        result_by_se_name = defaultdict(list) #deduplicate results
        for morphology, text, se_name in res:
            morphology = morphology.lower()
            if morphology not in result:
                result[morphology] = []
            if text not in result_by_se_name[se_name] and text.count(morphology)==1:
                result_by_se_name[se_name].append(text)
                result[morphology].append(text)
            else:
                print("filtering", text, se_name)
        return result

    @staticmethod
    def get_stanza_nlp(source_lang):
        global _nlp_stanza
        if _nlp_stanza is None:
            import stanza
            stanza_map = {
                'jp': 'ja'
            }
            stanza_lang_name = stanza_map.get(source_lang, source_lang)
            try:
                _nlp_stanza = stanza.Pipeline(stanza_lang_name, download_method=None, processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True, use_gpu=True)
            except (stanza.pipeline.core.LanguageNotDownloadedError, stanza.resources.common.ResourcesFileNotFoundError, FileNotFoundError):
                print(f'Downloading stanza model for {stanza_lang_name}...')
                stanza.download(stanza_lang_name)
                print(f'Stanza model for {stanza_lang_name} downloaded')
                _nlp_stanza = stanza.Pipeline(stanza_lang_name, download_method=None, processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True, use_gpu=True)
        return _nlp_stanza

    @staticmethod
    def preprocess_sentences(sentences, morphology, source_lang):
        result = []
        pattern = r'[^\s\w]*(' + re.escape(morphology) + r')[^\s\w]*'
        for sentence in sentences:
            if len(sentence.split()) >= 3:
                sentence = sentence.replace('\r', ' ').replace('\u2005', ' ')
                if source_lang == 'jp':
                    processed_sentence = sentence.replace(' ', '')
                else:
                    removed_trailing_char = re.sub(pattern, r'\1', sentence)
                    no_repeated_whitespace = re.sub(r'\s+', ' ', removed_trailing_char).strip()
                    processed_sentence = ' '.join(
                        ['<ROOT>' if re.sub(r'[^\w]', '', word.lower()) == morphology.lower() else word for word in no_repeated_whitespace.split()])
                    if not '<ROOT>' in processed_sentence or processed_sentence.count('<ROOT>') != 1:
                        print("morphology",morphology)
                        print("sentence",sentence)
                        print("processed_sentence",processed_sentence)
                        continue
                result.append(processed_sentence)
        return result

    @staticmethod
    def preprocess_stanza(sentences, source_lang):
        def process_string(string, initial_offset=0):
            doc = nlp(string)
            pos_sentence = []
            dependencies = []
            offset = initial_offset
            for stanza_sentence in doc.sentences:
                for word in stanza_sentence.words:
                    adjusted_id = word.id + offset
                    adjusted_head = word.head + offset if word.head != 0 else 0  # Root has head 0
                    if word.pos != 'PUNCT':
                        pos_sentence.append(
                            f'<{adjusted_id}:{word.pos}/{word.deprel}>({word.text})'
                        )
                    if word.deprel != 'punct':
                        dependencies.append((adjusted_id, adjusted_head, word.deprel))
                # Update the offset after each sentence
                offset += len(stanza_sentence.words)
            return pos_sentence, dependencies, offset

        nlp = Loader.get_stanza_nlp(source_lang)
        result = []
        for sentence in sentences:
            pos_sentence = []
            dependencies = []
            # print("sentence.split('<ROOT>')", sentence.split('<ROOT>'))
            if '<ROOT>' in sentence:
                before_root, after_root = sentence.split('<ROOT>')
                # Process before_root with initial offset 0
                pos_before_root, deps_before_root, offset = process_string(before_root)
                # Process after_root with the offset from before_root
                pos_after_root, deps_after_root, _ = process_string(after_root, initial_offset=offset)
                pos_sentence.extend(pos_before_root)
                pos_sentence.append('<ROOT>')
                pos_sentence.extend(pos_after_root)
                dependencies.extend(deps_before_root)
                dependencies.extend(deps_after_root)
            else:
                # If there's no '<ROOT>' in the sentence, process it as is
                pos_sentence, dependencies, _ = process_string(sentence)
            result.append({'sentence': ' '.join(pos_sentence), 'dependencies': dependencies})
        return result

    @staticmethod
    def load_sentences(sentences_dict, source_lang, preprocessor = None):
        processed_sentences_dict = {}
        for morphology, sentences in sentences_dict.items():
            sentences = Loader.preprocess_sentences(sentences, morphology, source_lang)
            if preprocessor=='stanza':
                sentences = Loader.preprocess_stanza(sentences, source_lang)
            processed_sentences_dict[morphology] = sentences
        return processed_sentences_dict

    @staticmethod
    def reduce_sentences(sentences_dict, n_sentences, seed=None):
        import random
        if seed is not None:
            random.seed(seed)
        all_sentences = [(key, sentence) for key, sentences in sentences_dict.items() for sentence in sentences]
        random.shuffle(all_sentences)
        all_sentences = all_sentences[:n_sentences]
        reduced_dict = defaultdict(list)
        for key, sentence in all_sentences:
            reduced_dict[key].append(sentence)
        return reduced_dict

    @staticmethod
    def load_sentences_from_word(word, source_lang, cache = True, use_cache = True, preprocessor = None, n_sentences = None):
        if n_sentences==0:
            return {}
        path = SENTENCES_DIR / (word + (f"_{preprocessor}" if preprocessor else '') + (f"_{n_sentences}" if n_sentences else '') + '.yaml')
        if use_cache:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    sentences_dict = yaml.load(f, Loader=yaml.FullLoader)
                    # sentences_dict = yaml.safe_load(f)
                    return OrderedDict(sentences_dict) if not n_sentences else Loader.reduce_sentences(sentences_dict, n_sentences)
        sentences_dict = Loader.load_db_sentences(word, source_lang)
        sentences_dict = sentences_dict if not n_sentences else Loader.reduce_sentences(sentences_dict, n_sentences)
        result = Loader.load_sentences(sentences_dict, source_lang, preprocessor=preprocessor)
        if cache:
            with open(SENTENCES_DIR / path, 'w') as f:
                yaml.dump(result, f)
        return OrderedDict(result)

    @staticmethod
    def extract_stanza_tags(sentence_info):
        sentence = sentence_info['sentence']
        dependencies = sentence_info['dependencies']
        # Regex to match <id:pos/deprel>(word)
        pattern = r'<(\d+):([^/]+)/([^>]+)>\(([^)]+)\)'

        pos_list = []
        deprel_list = []
        word_list = []

        for match in re.finditer(pattern, sentence):
            word_id, pos, deprel, word = match.groups()
            pos_list.append(f"<{pos}>")
            deprel_list.append(f"<{deprel}>")
            word_list.append(word)

        # Join the lists back into strings
        pos_string = " ".join(pos_list)
        deprel_string = " ".join(deprel_list)
        word_string = " ".join(word_list)

        return pos_string, deprel_string, word_string, dependencies