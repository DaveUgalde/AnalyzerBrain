"""
TextProcessing - Utilidades para procesamiento y análisis de texto.
Incluye limpieza, tokenización, normalización, extracción de entidades y comparación.
"""

import re
import string
import unicodedata
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from rapidfuzz import fuzz
import langdetect
from datetime import datetime
import logging
from ..core.exceptions import BrainException

logger = logging.getLogger(__name__)

class TextProcessing:
    """
    Utilidades avanzadas para procesamiento de texto.
    
    Características:
    1. Limpieza y normalización de texto
    2. Tokenización en múltiples niveles
    3. Extracción de entidades y conceptos
    4. Resumen y análisis de texto
    5. Detección de idioma y comparación
    """
    
    # Cache para modelos cargados
    _nlp_models = {}
    _nltk_resources_initialized = False
    
    @staticmethod
    def clean_text(
        text: str,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_mentions: bool = True,
        normalize_unicode: bool = True
    ) -> str:
        """
        Limpia y normaliza texto según criterios especificados.
        
        Args:
            text: Texto a limpiar
            remove_punctuation: Eliminar signos de puntuación
            remove_numbers: Eliminar números
            remove_stopwords: Eliminar palabras vacías (stopwords)
            lowercase: Convertir a minúsculas
            strip_whitespace: Normalizar espacios en blanco
            remove_html: Eliminar etiquetas HTML
            remove_urls: Eliminar URLs
            remove_emails: Eliminar direcciones email
            remove_mentions: Eliminar menciones (@usuario)
            normalize_unicode: Normalizar caracteres Unicode
            
        Returns:
            Texto limpio y normalizado
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Normalizar Unicode (NFKC)
        if normalize_unicode:
            cleaned = unicodedata.normalize('NFKC', cleaned)
        
        # Eliminar HTML
        if remove_html:
            cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        
        # Eliminar URLs
        if remove_urls:
            cleaned = re.sub(
                r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'
                r'([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
                ' ',
                cleaned
            )
        
        # Eliminar emails
        if remove_emails:
            cleaned = re.sub(
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                ' ',
                cleaned
            )
        
        # Eliminar menciones
        if remove_mentions:
            cleaned = re.sub(r'@\w+', ' ', cleaned)
        
        # Eliminar números
        if remove_numbers:
            cleaned = re.sub(r'\d+', ' ', cleaned)
        
        # Eliminar puntuación
        if remove_punctuation:
            # Mantener algunos símbolos útiles para código
            punctuation_to_keep = set('._-')
            translator = str.maketrans(
                '', '', 
                string.punctuation.translate(str.maketrans('', '', ''.join(punctuation_to_keep)))
            )
            cleaned = cleaned.translate(translator)
        
        # Convertir a minúsculas
        if lowercase:
            cleaned = cleaned.lower()
        
        # Eliminar stopwords
        if remove_stopwords:
            TextProcessing._ensure_nltk_resources()
            stop_words = set(stopwords.words('english'))
            tokens = cleaned.split()
            cleaned = ' '.join([word for word in tokens if word not in stop_words])
        
        # Normalizar espacios en blanco
        if strip_whitespace:
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    @staticmethod
    def tokenize_text(
        text: str,
        method: str = 'word',
        language: str = 'english',
        preserve_case: bool = False,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        stem: bool = False,
        lemmatize: bool = False
    ) -> List[str]:
        """
        Tokeniza texto en unidades significativas.
        
        Args:
            text: Texto a tokenizar
            method: 'word', 'sentence', 'char', 'subword'
            language: Idioma para modelos lingüísticos
            preserve_case: Mantener mayúsculas/minúsculas
            remove_punctuation: Eliminar tokens de puntuación
            remove_stopwords: Eliminar stopwords
            stem: Aplicar stemming
            lemmatize: Aplicar lematización
            
        Returns:
            Lista de tokens
        """
        if not text:
            return []
        
        TextProcessing._ensure_nltk_resources()
        
        # Preprocesamiento básico
        if not preserve_case:
            text = text.lower()
        
        tokens = []
        
        if method == 'sentence':
            # Tokenización por oraciones
            tokens = sent_tokenize(text, language=language)
        
        elif method == 'word':
            # Tokenización por palabras
            tokens = word_tokenize(text, language=language)
            
            # Filtrar puntuación
            if remove_punctuation:
                tokens = [token for token in tokens 
                         if token not in string.punctuation]
            
            # Eliminar stopwords
            if remove_stopwords:
                stop_words = set(stopwords.words(language))
                tokens = [token for token in tokens 
                         if token.lower() not in stop_words]
            
            # Stemming
            if stem:
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(token) for token in tokens]
            
            # Lemmatización
            if lemmatize:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        elif method == 'char':
            # Tokenización por caracteres
            tokens = list(text)
            
            if remove_punctuation:
                tokens = [char for char in tokens 
                         if char not in string.punctuation and not char.isspace()]
        
        elif method == 'subword':
            # Tokenización subword (BPE-like)
            # Implementación simple basada en espacios y caracteres especiales
            tokens = re.findall(r'\b\w+\b|[\w\']+|[^\w\s]', text)
        
        return tokens
    
    @staticmethod
    def normalize_text(
        text: str,
        form: str = 'NFKC',
        case: str = 'lower',
        whitespace: str = 'single',
        preserve_acronyms: bool = True
    ) -> str:
        """
        Normaliza texto de manera consistente.
        
        Args:
            text: Texto a normalizar
            form: Forma de normalización Unicode ('NFC', 'NFD', 'NFKC', 'NFKD')
            case: 'lower', 'upper', 'title', 'sentence', 'preserve'
            whitespace: 'single', 'collapse', 'trim', 'preserve'
            preserve_acronyms: Preservar mayúsculas en acrónimos
            
        Returns:
            Texto normalizado
        """
        if not text:
            return ""
        
        normalized = text
        
        # Normalización Unicode
        if form in ['NFC', 'NFD', 'NFKC', 'NFKD']:
            normalized = unicodedata.normalize(form, normalized)
        
        # Manejo de caso
        if case == 'lower':
            if preserve_acronyms:
                # Preservar acrónimos (palabras totalmente en mayúsculas)
                words = normalized.split()
                normalized_words = []
                for word in words:
                    if word.isupper() and len(word) > 1:
                        normalized_words.append(word)
                    else:
                        normalized_words.append(word.lower())
                normalized = ' '.join(normalized_words)
            else:
                normalized = normalized.lower()
        elif case == 'upper':
            normalized = normalized.upper()
        elif case == 'title':
            normalized = normalized.title()
        elif case == 'sentence':
            # Capitalización tipo oración
            sentences = re.split(r'([.!?]+\s*)', normalized)
            normalized = ''.join([
                sent.capitalize() if i % 2 == 0 else sent
                for i, sent in enumerate(sentences)
            ])
        
        # Manejo de espacios en blanco
        if whitespace == 'single':
            normalized = re.sub(r'\s+', ' ', normalized).strip()
        elif whitespace == 'collapse':
            normalized = re.sub(r'\s+', '', normalized)
        elif whitespace == 'trim':
            normalized = normalized.strip()
        # 'preserve' no hace nada
        
        return normalized
    
    @staticmethod
    def extract_entities(
        text: str,
        entity_types: List[str] = None,
        language: str = 'en',
        use_spacy: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extrae entidades nombradas del texto.
        
        Args:
            text: Texto de entrada
            entity_types: Tipos de entidades a extraer (PERSON, ORG, LOC, etc.)
            language: Código de idioma
            use_spacy: Usar spaCy para mejores resultados
            
        Returns:
            Lista de diccionarios con entidades encontradas
        """
        if not text:
            return []
        
        entities = []
        
        if use_spacy:
            try:
                nlp = TextProcessing._load_spacy_model(language)
                doc = nlp(text)
                
                for ent in doc.ents:
                    if entity_types is None or ent.label_ in entity_types:
                        entities.append({
                            'text': ent.text,
                            'type': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 1.0,  # spaCy no proporciona confianza
                            'source': 'spacy'
                        })
                        
            except Exception as e:
                logger.warning(f"spaCy entity extraction failed: {e}")
                # Fallback a regex básico
                entities.extend(TextProcessing._extract_entities_regex(text))
        else:
            entities.extend(TextProcessing._extract_entities_regex(text))
        
        return entities
    
    @staticmethod
    def summarize_text(
        text: str,
        max_sentences: int = 3,
        max_words: int = 100,
        language: str = 'english',
        algorithm: str = 'frequency'
    ) -> str:
        """
        Genera un resumen del texto.
        
        Args:
            text: Texto a resumir
            max_sentences: Máximo de oraciones en el resumen
            max_words: Máximo de palabras en el resumen
            language: Idioma del texto
            algorithm: Algoritmo de resumen ('frequency', 'textrank', 'luhn')
            
        Returns:
            Texto resumido
        """
        if not text:
            return ""
        
        TextProcessing._ensure_nltk_resources()
        
        # Tokenizar en oraciones
        sentences = sent_tokenize(text, language=language)
        
        if len(sentences) <= max_sentences:
            return text
        
        if algorithm == 'frequency':
            return TextProcessing._summarize_frequency(
                text, sentences, max_sentences, language
            )
        elif algorithm == 'textrank':
            return TextProcessing._summarize_textrank(
                text, sentences, max_sentences, language
            )
        elif algorithm == 'luhn':
            return TextProcessing._summarize_luhn(
                text, sentences, max_sentences, language
            )
        else:
            raise ValueError(f"Unknown summarization algorithm: {algorithm}")
    
    @staticmethod
    def detect_language(
        text: str,
        reliable_only: bool = True
    ) -> Dict[str, Any]:
        """
        Detecta el idioma del texto.
        
        Args:
            text: Texto a analizar
            reliable_only: Solo retornar si la detección es confiable
            
        Returns:
            Diccionario con información del idioma detectado
        """
        if not text or len(text.strip()) < 10:
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'reliable': False,
                'text_sample': text
            }
        
        try:
            from langdetect import detect, detect_langs, LangDetectException
            
            # Obtener todos los idiomas posibles con probabilidades
            languages = detect_langs(text)
            
            if not languages:
                raise LangDetectException("No languages detected")
            
            # Idioma principal
            main_lang = languages[0]
            
            # Verificar confiabilidad
            is_reliable = (
                main_lang.prob > 0.8 and 
                (len(languages) == 1 or languages[1].prob < main_lang.prob * 0.5)
            )
            
            if reliable_only and not is_reliable:
                return {
                    'language': 'unknown',
                    'confidence': main_lang.prob,
                    'reliable': False,
                    'candidates': [
                        {'lang': str(lang.lang), 'prob': lang.prob}
                        for lang in languages[:3]
                    ],
                    'text_sample': text[:100]
                }
            
            return {
                'language': str(main_lang.lang),
                'confidence': main_lang.prob,
                'reliable': is_reliable,
                'candidates': [
                    {'lang': str(lang.lang), 'prob': lang.prob}
                    for lang in languages[:3]
                ],
                'text_sample': text[:100]
            }
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            
            # Fallback simple basado en caracteres comunes
            return TextProcessing._detect_language_simple(text)
    
    @staticmethod
    def compare_texts(
        text1: str,
        text2: str,
        method: str = 'cosine',
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Compara dos textos y calcula similitudes.
        
        Args:
            text1: Primer texto
            text2: Segundo texto
            method: Método de comparación ('cosine', 'jaccard', 'levenshtein', 'fuzzy')
            preprocess: Aplicar preprocesamiento antes de comparar
            
        Returns:
            Diccionario con métricas de similitud
        """
        if preprocess:
            text1 = TextProcessing.clean_text(text1)
            text2 = TextProcessing.clean_text(text2)
        
        if method == 'cosine':
            similarity = TextProcessing._cosine_similarity(text1, text2)
        elif method == 'jaccard':
            similarity = TextProcessing._jaccard_similarity(text1, text2)
        elif method == 'levenshtein':
            similarity = TextProcessing._levenshtein_similarity(text1, text2)
        elif method == 'fuzzy':
            similarity = TextProcessing._fuzzy_similarity(text1, text2)
        else:
            raise ValueError(f"Unknown comparison method: {method}")
        
        return {
            'method': method,
            'similarity': similarity,
            'text1_length': len(text1),
            'text2_length': len(text2),
            'text1_preview': text1[:100],
            'text2_preview': text2[:100],
            'comparison_timestamp': datetime.now()
        }
    
    # ========== MÉTODOS PRIVADOS ==========
    
    @staticmethod
    def _ensure_nltk_resources() -> None:
        """Asegura que los recursos de NLTK estén disponibles."""
        if not TextProcessing._nltk_resources_initialized:
            try:
                nltk_resources = [
                    'punkt',
                    'stopwords',
                    'wordnet',
                    'averaged_perceptron_tagger'
                ]
                
                for resource in nltk_resources:
                    try:
                        nltk.data.find(f'tokenizers/{resource}')
                    except LookupError:
                        nltk.download(resource, quiet=True)
                
                TextProcessing._nltk_resources_initialized = True
                logger.debug("NLTK resources initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize NLTK resources: {e}")
                raise BrainException(f"NLTK resources unavailable: {e}")
    
    @staticmethod
    def _load_spacy_model(language: str = 'en'):
        """Carga modelo spaCy con cache."""
        model_key = f"spacy_{language}"
        
        if model_key not in TextProcessing._nlp_models:
            try:
                if language == 'en':
                    import spacy
                    TextProcessing._nlp_models[model_key] = spacy.load("en_core_web_sm")
                else:
                    # Intentar cargar modelo específico
                    import spacy
                    TextProcessing._nlp_models[model_key] = spacy.load(f"{language}_core_news_sm")
            except Exception as e:
                logger.warning(f"Could not load spaCy model for {language}: {e}")
                # Usar modelo pequeño en inglés como fallback
                import spacy
                TextProcessing._nlp_models[model_key] = spacy.load("en_core_web_sm")
        
        return TextProcessing._nlp_models[model_key]
    
    @staticmethod
    def _extract_entities_regex(text: str) -> List[Dict[str, Any]]:
        """Extrae entidades usando expresiones regulares básicas."""
        entities = []
        
        # Patrones para diferentes tipos de entidades
        patterns = {
            'EMAIL': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'URL': r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
            'PHONE': r'(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}',
            'DATE': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'NUMBER': r'\b\d+\b',
        }
        
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7,
                    'source': 'regex'
                })
        
        return entities
    
    @staticmethod
    def _summarize_frequency(text, sentences, max_sentences, language):
        """Resumen basado en frecuencia de palabras."""
        # Tokenizar palabras y eliminar stopwords
        stop_words = set(stopwords.words(language))
        words = word_tokenize(text.lower())
        words = [word for word in words 
                 if word not in stop_words and word not in string.punctuation]
        
        # Calcular frecuencia de palabras
        word_freq = Counter(words)
        
        # Ponderar oraciones por frecuencia de palabras
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            score = sum(word_freq[word] for word in sentence_words 
                       if word in word_freq)
            sentence_scores[i] = score
        
        # Seleccionar las mejores oraciones
        ranked_sentences = sorted(
            sentence_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_sentences]
        
        # Ordenar por posición original
        summary_sentences = [sentences[idx] for idx, _ in 
                            sorted(ranked_sentences, key=lambda x: x[0])]
        
        return ' '.join(summary_sentences)
    
    @staticmethod
    def _summarize_textrank(text, sentences, max_sentences, language):
        """Implementación simple de TextRank."""
        # Tokenizar palabras por oración
        sentence_tokens = [word_tokenize(sent.lower()) for sent in sentences]
        
        # Crear matriz de similitud
        n = len(sentences)
        similarity_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    set_i = set(sentence_tokens[i])
                    set_j = set(sentence_tokens[j])
                    if len(set_i | set_j) > 0:
                        similarity = len(set_i & set_j) / len(set_i | set_j)
                        similarity_matrix[i][j] = similarity
        
        # Pagerank simple
        scores = [1.0] * n
        damping = 0.85
        iterations = 20
        
        for _ in range(iterations):
            new_scores = [0.0] * n
            for i in range(n):
                rank_sum = 0.0
                for j in range(n):
                    if similarity_matrix[j][i] > 0:
                        rank_sum += similarity_matrix[j][i] * scores[j]
                new_scores[i] = (1 - damping) + damping * rank_sum
            scores = new_scores
        
        # Seleccionar mejores oraciones
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in ranked[:max_sentences]]
        top_indices.sort()
        
        return ' '.join([sentences[i] for i in top_indices])
    
    @staticmethod
    def _summarize_luhn(text, sentences, max_sentences, language):
        """Algoritmo de Luhn para resumen."""
        # Implementación simplificada
        return TextProcessing._summarize_frequency(
            text, sentences, max_sentences, language
        )
    
    @staticmethod
    def _detect_language_simple(text: str) -> Dict[str, Any]:
        """Detección simple de idioma basada en caracteres comunes."""
        # Frecuencias aproximadas de caracteres por idioma
        language_patterns = {
            'en': set('etaoinshrdlcumwfgypbvkjxqz'),
            'es': set('eaosrnidltcmupbgvyqfhzjñxkáéíóúü'),
            'fr': set('esaitnrulodcmpévqfbghjàxèyêzôùîçœ'),
            'de': set('enisratdhulcgmobwfkzvpüäßjöyqx'),
        }
        
        text_lower = text.lower()
        text_chars = set(text_lower)
        
        scores = {}
        for lang, chars in language_patterns.items():
            common = len(text_chars & chars)
            total = len(text_chars)
            if total > 0:
                scores[lang] = common / total
        
        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            return {
                'language': best_lang[0],
                'confidence': best_lang[1],
                'reliable': best_lang[1] > 0.7,
                'text_sample': text[:100]
            }
        
        return {
            'language': 'unknown',
            'confidence': 0.0,
            'reliable': False,
            'text_sample': text[:100]
        }
    
    @staticmethod
    def _cosine_similarity(text1: str, text2: str) -> float:
        """Calcula similitud coseno entre dos textos."""
        # Tokenizar
        tokens1 = TextProcessing.tokenize_text(text1, remove_stopwords=True)
        tokens2 = TextProcessing.tokenize_text(text2, remove_stopwords=True)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Vocabulario unificado
        vocab = set(tokens1 + tokens2)
        
        # Vectores de frecuencia
        vec1 = Counter(tokens1)
        vec2 = Counter(tokens2)
        
        # Calcular producto punto
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in vocab)
        
        # Calcular magnitudes
        mag1 = sum(val ** 2 for val in vec1.values()) ** 0.5
        mag2 = sum(val ** 2 for val in vec2.values()) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Calcula similitud de Jaccard entre dos textos."""
        set1 = set(TextProcessing.tokenize_text(text1, remove_stopwords=True))
        set2 = set(TextProcessing.tokenize_text(text2, remove_stopwords=True))
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _levenshtein_similarity(text1: str, text2: str) -> float:
        """Calcula similitud basada en distancia de Levenshtein."""
        # Implementación de distancia de Levenshtein
        if len(text1) < len(text2):
            text1, text2 = text2, text1
        
        if len(text2) == 0:
            return 0.0
        
        previous_row = range(len(text2) + 1)
        
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        distance = previous_row[-1]
        max_len = max(len(text1), len(text2))
        
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    @staticmethod
    def _fuzzy_similarity(text1: str, text2: str) -> float:
        """Calcula similitud usando fuzzy matching."""
        try:
            from rapidfuzz import fuzz
            ratio = fuzz.ratio(text1, text2)
            return ratio / 100.0
        except ImportError:
            # Fallback a Levenshtein
            return TextProcessing._levenshtein_similarity(text1, text2)