import yaml
import re
from melon_clustering import SENTENCES_DIR
import stanza
from pathlib import Path
from collections import defaultdict

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
            if text not in result_by_se_name[se_name]:
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
                _nlp_stanza = stanza.Pipeline(stanza_lang_name, download_method=None, processors='tokenize,pos,lemma', use_gpu=True)
            except (stanza.pipeline.core.LanguageNotDownloadedError, stanza.resources.common.ResourcesFileNotFoundError, FileNotFoundError):
                print(f'Downloading stanza model for {stanza_lang_name}...')
                stanza.download(stanza_lang_name)
                print(f'Stanza model for {stanza_lang_name} downloaded')
                _nlp_stanza = stanza.Pipeline(stanza_lang_name, download_method=None, processors='tokenize,pos,lemma', use_gpu=True)
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
                    if not '<ROOT>' in processed_sentence:
                        print("morphology",morphology)
                        print("sentence",sentence)
                        print("processed_sentence",processed_sentence)
                        continue
                result.append(processed_sentence)
        return result

    @staticmethod
    def preprocess_stanza(sentences, source_lang):
        def process_string(string):
            doc = nlp(string)
            pos_sentence = []
            for stanza_sentence in doc.sentences:
                for word in stanza_sentence.words:
                    if word.pos!='PUNCT':
                        pos_sentence.append(f'<{word.pos}/{word.xpos}>({word.text})')
            return pos_sentence

        nlp = Loader.get_stanza_nlp(source_lang)
        result = []
        for sentence in sentences:
            pos_sentence = []
            pos_sentence.extend(process_string(sentence.split('<ROOT>')[0]))
            pos_sentence.append('<ROOT>')
            pos_sentence.extend(process_string(sentence.split('<ROOT>')[1]))
            result.append(' '.join(pos_sentence))
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
    def load_sentences_from_word(word, source_lang, cache = True, use_cache = True, preprocessor = None):
        path = SENTENCES_DIR / (word + (f"_{preprocessor}" if preprocessor else '') + '.yaml')
        if use_cache:
            if path.exists():
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
        sentences_dict = Loader.load_db_sentences(word, source_lang)
        result = Loader.load_sentences(sentences_dict, source_lang, preprocessor=preprocessor)
        if cache:
            with open(SENTENCES_DIR / path, 'w') as f:
                yaml.dump(result, f)
        return result
