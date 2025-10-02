import re
from typing import List
import logging

logger = logging.getLogger(__name__)


# TODO: do we need that at all? during debugging I can't see any difference before/after
class MarkdownCleaner:

    def clean_markdown_file(self, md_path: str) -> str:
        """Cleanup markdown files"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned_content = self._proces_cleanup(content)
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            logger.info(f"Cleaned markdown file: {md_path}")
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Error cleaning markdown file {md_path}: {e}")
            return ""
    
    def _proces_cleanup(self, content: str) -> str:
        """Clean content step by step (skip code blocks)"""
        
        parts = self._split_code_blocks(content)
        cleaned_parts = []
        
        for is_code, text in parts:
            if is_code:
                # leave code blocks untouched
                cleaned_parts.append(text)
            else:
                # clean text blocks
                text = self._remove_emojis(text)
                text = self._fix_whitespace(text)
                text = self._remove_formatting_artifacts(text)
                text = self._fix_markdown_syntax(text)
                text = self._remove_garbage_characters(text)
                cleaned_parts.append(text)
        
        return "".join(cleaned_parts)

    def _split_code_blocks(self, content: str) -> List[tuple]:
        """Split markdown into (is_code, text) parts"""
        parts = []
        pattern = re.compile(r'(```.*?```)', re.DOTALL)
        last = 0
        for m in pattern.finditer(content):
            if m.start() > last:
                parts.append((False, content[last:m.start()]))
            parts.append((True, m.group(0)))
            last = m.end()
        if last < len(content):
            parts.append((False, content[last:]))
        return parts

    def _remove_emojis(self, content: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub('', content)
    
    def _fix_whitespace(self, content: str) -> str:
        lines = content.splitlines()
        fixed_lines = []
        for line in lines:
            if "|" in line and re.match(r'^\s*\|.*\|\s*$', line):
                # looks like a table row, leave spacing as is
                fixed_lines.append(line)
            else:
                line = re.sub(r'[ \t]{2,}', ' ', line)  # 2+ spaces → 1
                fixed_lines.append(line)
        content = "\n".join(fixed_lines)
        # collapse 3+ newlines to 2
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content
    
    def _remove_formatting_artifacts(self, content: str) -> str:
        # Replace 4+ asterisks with 3 (preserve *** bold+italic)
        content = re.sub(r'\*{4,}', '***', content)
        # Replace 4+ underscores with 3
        content = re.sub(r'_{4,}', '___', content)
        # Replace 4+ dashes with 3
        content = re.sub(r'-{4,}', '---', content)
        return content
    
    def _fix_markdown_syntax(self, content: str) -> str:
        # Ensure space after headers
        content = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', content, flags=re.MULTILINE)
        return content
    
    def _remove_garbage_characters(self, content: str) -> str:
        # remove control chars except newline/tab
        content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)
        content = re.sub(r'\.{5,}', '...', content)   # 5+ dots -> ...
        content = re.sub(r'!{5,}', '!!!', content)   # 5+ ! -> !!!
        content = re.sub(r'\?{5,}', '???', content)  # 5+ ? -> ???
        return content
