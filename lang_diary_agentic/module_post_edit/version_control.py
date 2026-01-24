import typing as ty
import difflib
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..models.generation_records import HistoryRecord
from pydantic import BaseModel


class DiaryVersionManager:
    
    @staticmethod
    def compute_text_diff(old_text: str, new_text: str) -> str:
        """
        Generates a Git-style Unified Diff between two strings.
        This string is sufficient to rollback changes later (using patch logic).
        """
        # difflib expects lists of lines
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        
        # Generate the diff
        # 'fromfile' and 'tofile' are just labels for the header
        diff_generator = difflib.unified_diff(
            old_lines, 
            new_lines, 
            fromfile='v_old', 
            tofile='v_new', 
            lineterm=''
        )
        
        # Convert the generator to a single string
        return "".join(diff_generator)

    def create_history_record_expression_filed(
        self,
        diary_id: str, 
        current_version: int,
        operation: str,
        primary_id: str,
        expression_original: ty.Optional[str],
        expression_translation: ty.Optional[str],
    ) -> Optional[HistoryRecord]:
        """
        Creates a unified history record. Returns None if no changes are detected.
        """
        assert operation in ('delete', 'add')
        obj_changes = {}

        obj_changes['primary_id'] = primary_id
        obj_changes['operation'] = operation
        obj_changes['expression_original'] = expression_original
        obj_changes['expression_translation'] = expression_translation
            
        return HistoryRecord(
            history_id=str(uuid.uuid4()),
            primary_id_DiaryEntry=diary_id,
            version_from=current_version,
            version_to=current_version + 1,
            created_at=datetime.now(),
            changes=obj_changes
        )

    def create_history_record_text_filed(
        self,
        diary_id: str, 
        current_version: int,
        field_name: str, 
        old_text: str, 
        new_text: str
        ) -> Optional[HistoryRecord]:
        """
        Creates a unified history record. Returns None if no changes are detected.
        """
        
        # 1. Container for the unified JSON
        combined_changes = {}
        has_changes = False

        # Check Text Diff
        # Only compute if texts are actually different (simple string check first for speed)
        if old_text != new_text:
            text_diff = DiaryVersionManager.compute_text_diff(old_text, new_text)
            if text_diff:
                combined_changes[field_name] = text_diff
                has_changes = True
            # end if
        # end if

        # 4. Return Record only if something changed
        if not has_changes:
            return None
            
        return HistoryRecord(
            history_id=str(uuid.uuid4()),
            primary_id_DiaryEntry=diary_id,
            version_from=current_version,
            version_to=current_version + 1,
            created_at=datetime.now(),
            changes=combined_changes
        )