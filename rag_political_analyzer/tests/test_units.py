# rag_political_analyzer/tests/test_units.py
import unittest
import spacy
import os
import json
import asyncio
from typing import Dict, Any

# Add project root to sys.path to allow imports from app
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.ingestion import extract_spacy_features
from app.core.retrieval import enrich_query_with_spacy
from app.agents.memory_agent import MemoryAgent, DEFAULT_MEMORY_DIR

# Ensure spaCy model is available for tests
try:
    nlp_for_tests = spacy.load("en_core_web_sm")
except OSError:
    print("Test setup: Downloading en_core_web_sm for spaCy...")
    spacy.cli.download("en_core_web_sm")
    nlp_for_tests = spacy.load("en_core_web_sm")


class TestCoreIngestion(unittest.TestCase):
    def setUp(self):
        self.nlp = nlp_for_tests # Use the preloaded model

    def test_extract_spacy_features_entities(self):
        text = "Apple Inc. is looking at buying U.K. startup for $1 billion."
        # Pass the nlp model directly as per the refactored extract_spacy_features
        features = extract_spacy_features(text, self.nlp)

        self.assertIn("entities", features)
        self.assertTrue(any(ent['text'] == "Apple Inc." and ent['label'] == "ORG" for ent in features['entities']))
        self.assertTrue(any(ent['text'] == "U.K." and ent['label'] == "GPE" for ent in features['entities']))
        self.assertTrue(any(ent['text'] == "$1 billion" and ent['label'] == "MONEY" for ent in features['entities']))

        self.assertIn("noun_chunks", features)
        self.assertIn("Apple Inc.", features["noun_chunks"])

        self.assertIn("keywords", features)
        self.assertIn("apple", features["keywords"]) # Lemmatized
        self.assertIn("startup", features["keywords"])
        self.assertNotIn("is", features["keywords"]) # Stop word

    def test_extract_spacy_features_empty_text(self):
        text = ""
        features = extract_spacy_features(text, self.nlp)
        self.assertEqual(features["entities"], [])
        self.assertEqual(features["noun_chunks"], [])
        self.assertEqual(features["keywords"], [])


class TestCoreRetrieval(unittest.TestCase):
    def setUp(self):
        self.nlp = nlp_for_tests

    def test_enrich_query_basic(self):
        query = "Tell me about economic policies in Germany"
        enriched = enrich_query_with_spacy(query, self.nlp)
        self.assertIn("economic policies", enriched)
        self.assertIn("germany", enriched.lower())
        self.assertIn("policy", enriched.lower())

    def test_enrich_query_no_new_terms(self):
        query = "Germany"
        enriched = enrich_query_with_spacy(query, self.nlp)
        self.assertEqual(query, enriched)

    def test_enrich_query_with_entities_and_nouns(self):
        query = "Impact of EU regulations on tech companies like Apple and Google"
        enriched = enrich_query_with_spacy(query, self.nlp)
        self.assertIn("EU regulations", enriched)
        self.assertIn("tech companies", enriched)
        self.assertIn("apple", enriched.lower())
        self.assertIn("google", enriched.lower())
        self.assertIn("regulation", enriched.lower())
        self.assertIn("company", enriched.lower())


class TestMemoryAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_ltm_file = os.path.join(DEFAULT_MEMORY_DIR, "_test_ltm.json")
        self.test_conv_file = os.path.join(DEFAULT_MEMORY_DIR, "_test_conv_hist.json")
        os.makedirs(DEFAULT_MEMORY_DIR, exist_ok=True)

        # Clear any previous test files before each test
        if os.path.exists(self.test_ltm_file): os.remove(self.test_ltm_file)
        if os.path.exists(self.test_conv_file): os.remove(self.test_conv_file)

        self.memory_agent = MemoryAgent(
            memory_file_path=self.test_ltm_file,
            conversation_memory_path=self.test_conv_file
        )

    def tearDown(self):
        if os.path.exists(self.test_ltm_file): os.remove(self.test_ltm_file)
        if os.path.exists(self.test_conv_file): os.remove(self.test_conv_file)

    async def test_store_and_retrieve_fact(self):
        fact_data = {"key": "test_fact", "value": "This is a test.", "category": "testing"}
        await self.memory_agent.execute({"type": "store_fact", "data": fact_data})

        result = await self.memory_agent.execute({"type": "retrieve_fact", "data": {"key": "test_fact", "category": "testing"}})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["fact"]["value"], "This is a test.")

    async def test_store_and_get_conversation_history(self):
        turn1 = {"role": "user", "content": "Hello"}
        turn2 = {"role": "assistant", "content": "Hi there"}
        await self.memory_agent.execute({"type": "store_conversation_turn", "data": turn1})
        await self.memory_agent.execute({"type": "store_conversation_turn", "data": turn2})

        result = await self.memory_agent.execute({"type": "get_conversation_history"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history"][0]["content"], "Hello")

        result_last_n = await self.memory_agent.execute({"type": "get_conversation_history", "data": {"last_n_turns": 1}})
        self.assertEqual(len(result_last_n["history"]), 1)
        self.assertEqual(result_last_n["history"][0]["content"], "Hi there")

    async def test_clear_conversation_history(self):
        await self.memory_agent.execute({"type": "store_conversation_turn", "data": {"role": "user", "content": "Hello"}})
        await self.memory_agent.execute({"type": "clear_conversation_history"})
        result = await self.memory_agent.execute({"type": "get_conversation_history"})
        self.assertEqual(len(result["history"]), 0)


if __name__ == '__main__':
    unittest.main()
```
