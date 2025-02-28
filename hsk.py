from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, Text, func
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timedelta
import random
import os

Base = declarative_base()

class ChineseWord(Base):
    __tablename__ = "chinese_words"
    
    id = Column(Integer, primary_key=True)
    simplified = Column(String, index=True)
    traditional = Column(String)
    radical = Column(String)
    level = Column(Integer, index=True)
    frequency = Column(Float)
    pos = Column(String)  # Part of speech
    pinyin = Column(String)
    meanings = Column(Text)
    correct_count = Column(Integer, default=0)
    incorrect_count = Column(Integer, default=0)
    examples = Column(Text, default="[]")
    
    # SRS fields
    srs_level = Column(Integer, default=0)
    next_review = Column(Date, default=datetime.now().date)
    last_reviewed = Column(Date, nullable=True)
    is_favorite = Column(Boolean, default=False)

class HSKManager:
    def __init__(self, db_path=None):
        # Use the DB_PATH from main.py if not specified
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', 'hsk.db')
        
        # Create directory for the database if it doesn't exist
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        
        # Create database engine and session
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.ChineseWord = ChineseWord
    
    def sample_words(self, count=5, min_level=1, max_level=None):
        """Sample words from the database, prioritizing unseen words."""
        if not max_level:
            max_level = min_level
            
        # Query for unseen words first
        unseen_words = self.session.query(ChineseWord).filter(
            ChineseWord.level.between(min_level, max_level),
            ChineseWord.correct_count == 0
        ).order_by(func.random()).limit(count).all()
        
        result = []
        for word in unseen_words:
            result.append({
                "id": word.id,
                "simplified": word.simplified,
                "traditional": word.traditional,
                "pinyin": word.pinyin,
                "meanings": word.meanings,
                "level": word.level
            })
        
        # If we need more words, sample from seen words
        if len(result) < count:
            remaining_count = count - len(result)
            seen_words = self.session.query(ChineseWord).filter(
                ChineseWord.level.between(min_level, max_level),
                ChineseWord.correct_count > 0,
                ~ChineseWord.id.in_([w["id"] for w in result])
            ).order_by(func.random()).limit(remaining_count).all()
            
            for word in seen_words:
                result.append({
                    "id": word.id,
                    "simplified": word.simplified,
                    "traditional": word.traditional,
                    "pinyin": word.pinyin,
                    "meanings": word.meanings,
                    "level": word.level
                })
        
        return result
    
    def update_words(self, transcription, sampled_words=None):
        """
        Check if specific words appear in the transcription and 
        update their correct_count accordingly.
        
        Args:
            transcription: The transcribed text from speech recognition
            sampled_words: List of word dictionaries to check, with at least 'id' key
                        If None, will check all words (legacy behavior)
        
        Returns:
            List of dictionaries with word update results
        """
        results = []
        print(sampled_words)
        
        # If specific words were provided, only check those
        if sampled_words and isinstance(sampled_words, list):
            # Extract word IDs
            word_ids = []
            for word in sampled_words:
                if isinstance(word, dict) and 'id' in word:
                    word_ids.append(word['id'])
            
            # Get only the specified words from database
            words_to_check = self.session.query(ChineseWord).filter(
                ChineseWord.id.in_(word_ids)
            ).all()
            
            print(f"Checking {len(words_to_check)} specific words against transcription")
        else:
            # Legacy behavior - check all words (not recommended for practice)
            words_to_check = self.session.query(ChineseWord).all()
            print(f"Warning: Checking all {len(words_to_check)} words against transcription")
        
        # Track which words were matched
        matched_words = []
        unmatched_words = []
        
        # Check each word against the transcription
        for word in words_to_check:
            simplified = word.simplified
            
            # Check if the word appears in the transcription
            is_correct = simplified in transcription
            
            if is_correct:
                # Update word stats for correct matches
                word.correct_count = (word.correct_count or 0) + 1
                word.last_reviewed = datetime.now().date()
                
                # Update SRS level and next review date
                word.srs_level = min(7, (word.srs_level or 0) + 1)
                
                days_until_review = self._get_interval(word.srs_level)
                word.next_review = (datetime.now() + timedelta(days=days_until_review)).date()
                
                matched_words.append(word)
            else:
                # Track words that weren't matched
                unmatched_words.append(word)
            
            # Add result for this word
            results.append({
                "id": word.id,
                "word": simplified,
                "correct": is_correct
            })
        
        # Commit changes to database
        if matched_words:
            self.session.commit()
            print(matched_words[0].srs_level)
            print(f"Updated {len(matched_words)} words in database")
        
        if unmatched_words:
            print(f"Words not found in transcription: {', '.join(w.simplified for w in unmatched_words)}")
        
        return results
    
    def update_word_result(self, word_id, was_correct):
        """Update a single word's status after practice."""
        word = self.session.query(ChineseWord).filter(ChineseWord.id == word_id).first()
        if not word:
            return {"error": f"Word with ID {word_id} not found"}
        
        # Update last reviewed date
        word.last_reviewed = datetime.now().date()
        
        if was_correct:
            # Increment correct count
            word.correct_count += 1
            
            # Update SRS level (increase)
            word.srs_level = min(7, word.srs_level + 1)
        else:
            # Increment incorrect count if field exists
            if hasattr(word, 'incorrect_count'):
                word.incorrect_count += 1
            
            # Update SRS level (decrease)
            word.srs_level = max(0, word.srs_level - 2)
        
        # Set next review date based on SRS level
        days_until_review = self._get_interval(word.srs_level)
        word.next_review = (datetime.now() + timedelta(days=days_until_review)).date()
        
        # Commit changes
        self.session.commit()
        
        return {
            "id": word.id,
            "word": word.simplified,
            "correct": was_correct,
            "srs_level": word.srs_level,
            "next_review": word.next_review.isoformat()
        }
    
    def get_unseen_words_by_level(self, level):
        """Get all unseen words at a specific HSK level."""
        return self.session.query(ChineseWord).filter(
            ChineseWord.level == level,
            ChineseWord.correct_count == 0
        ).all()
    
    def get_words_due_for_review(self, count=10, levels=None):
        """Get words that are due for review based on SRS schedule."""
        query = self.session.query(ChineseWord).filter(
            ChineseWord.next_review <= datetime.now().date()
        )
        
        if levels:
            query = query.filter(ChineseWord.level.in_(levels))
        
        # Order by words that have been waiting longest
        words = query.order_by(ChineseWord.next_review).limit(count).all()
        
        result = []
        for word in words:
            result.append({
                "id": word.id,
                "simplified": word.simplified,
                "traditional": word.traditional,
                "pinyin": word.pinyin,
                "meanings": word.meanings,
                "level": word.level,
                "srs_level": word.srs_level,
                "next_review": word.next_review.isoformat() if word.next_review else None
            })
        
        return result
    
    def get_all_words(self):
        """Get all words in the database."""
        return self.session.query(ChineseWord).all()
    
    def get_word_by_id(self, word_id):
        """Get a word by its ID."""
        return self.session.query(ChineseWord).filter(ChineseWord.id == word_id).first()
    
    def reset_all_words(self, level=None):
        """
        Reset all words to their initial learning state.
        
        Args:
            level: Optional HSK level to reset. If None, resets all words.
            
        Returns:
            Number of words reset
        """
        query = self.session.query(ChineseWord)
        
        if level is not None:
            query = query.filter(ChineseWord.level == level)
        
        # Get all matching words
        words = query.all()
        reset_count = 0
        
        # Reset each word's learning properties
        for word in words:
            word.correct_count = 0
            word.incorrect_count = 0
            word.srs_level = 0
            word.next_review = datetime.now().date()  # Due immediately
            word.last_reviewed = None
            # Don't reset favorite status as that's user preference
            reset_count += 1
        
        # Save changes to database
        self.session.commit()
        
        print(f"Reset {reset_count} words to initial state")
        return reset_count
        
    def remove_words_from_file(self, filename):
        """Read a text file containing a list of words and remove them from the database."""
        with open(filename, 'r', encoding='utf-8') as f:
            words_to_remove = [line.strip() for line in f if line.strip()]
        
        for word in words_to_remove:
            self.session.query(ChineseWord).filter(ChineseWord.simplified == word).delete()
        
        self.session.commit()
    
    def toggle_favorite(self, word_id):
        """Toggle a word's favorite status."""
        word = self.session.query(ChineseWord).filter(ChineseWord.id == word_id).first()
        if word:
            word.is_favorite = not word.is_favorite
            self.session.commit()
            return {"id": word_id, "is_favorite": word.is_favorite}
        return {"error": "Word not found"}
    
    def get_favorites(self):
        """Get all favorite words."""
        return self.session.query(ChineseWord).filter(ChineseWord.is_favorite == True).all()
    
    def _get_interval(self, srs_level):
        """Get the number of days until next review based on SRS level."""
        # Spaced repetition intervals (in days)
        intervals = [1, 3, 7, 14, 30, 60, 120, 240]
        return intervals[min(srs_level, len(intervals) - 1)]
    
    def __del__(self):
        """Close the session when the object is destroyed."""
        self.session.close()

# For testing
if __name__ == "__main__":
    manager = HSKManager()
    
    # Sample usage
    # print("Sampling 5 HSK level 1 words:")
    # words = manager.sample_words(5, 1)
    # for word in words:
    #     print(f"{word['simplified']} ({word['pinyin']}): {word['meanings']}")
    
    # # Test update_words
    # print("\nUpdating words with transcription '我喜欢学习中文':")
    # results = manager.update_words("我喜欢学习中文")
    # for result in results:
    #     print(f"{result['word']}: {'Correct' if result['correct'] else 'Incorrect'}")
    
    # # Test get_words_due_for_review
    # print("\nWords due for review:")
    # due_words = manager.get_words_due_for_review(5)
    # for word in due_words:
    #     print(f"{word['simplified']} (Next review: {word['next_review']})")
    #manager.reset_all_words()
    import json
    words = manager.get_all_words()
    for word in words:
        #print(json.loads(word.examples)[0]["simplified"])
        try:
            s = json.loads(word.examples)[0]["simplified"]
            p = json.loads(word.examples)[0]["pinyin"]
            e = json.loads(word.examples)[0]["english"]
        except:
            print(word.id, word.simplified, word.level)
            print(json.loads(word.examples))
            break