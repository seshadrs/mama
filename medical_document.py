from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum
from typing import List, Optional
import pickle


class MedicalDocument:

    class Parse(BaseModel):
        class Type(Enum):
            DOCTOR_NOTES = "doctor_notes"
            LAB_RESULTS = "lab_results"
            AFTER_VISIT_SUMMARY = "after_visit_summary"
            MESSAGE = "message"
            OTHER = "other"

        type: Optional[Type] = Field(description="The type of the medical document.")
        clean_content: Optional[str] = Field(
            description="The content of the medical document devoid of any metadata, headers, footers, extra spaces, extraneous text, etc. Retain names of people."
        )
        summary: Optional[str] = Field(
            description="A summary of the medical document in under 20 words."
        )
        medical_specialty: Optional[str] = Field(
            description="The medical specialty of the document."
        )
        medical_conditions: Optional[List[str]] = Field(
            description="A list of medical conditions mentioned in the document."
        )
        medical_professional_name: Optional[str] = Field(
            description="Name of the medical professional who is the author or recipient of the document."
        )
        medical_institution_name: Optional[str] = Field(
            description="Name of the medical institution."
        )
        year: Optional[int] = Field(
            description="Original Year in which the medical document was authored. Use the earliest date in the document."
        )
        month: Optional[int] = Field(
            description="Original Month in which the medical document was authored. Use the earliest date in the document."
        )
        day: Optional[int] = Field(
            description="Original Day in which the medical document was authored. Use the earliest date in the document."
        )

    def __init__(self, pages, medical_document_parser):
        assert len(pages) > 0, "No pages found"
        self.pages = pages
        self.filepath = self.pages[0].metadata["source"]
        self.raw_content = " ".join([page.page_content for page in self.pages])
        self.parse = medical_document_parser.invoke(self.raw_content)
        self.date = f"{self.parse.year or 0}-{(self.parse.month or 0):02d}-{(self.parse.day or 0):02d}"

    def __lt__(self, other):
        if not isinstance(other, MedicalDocument):
            return NotImplemented

        return self.date < other.date

    def __str__(self):
        return f"""DATE: {self.date}
        TYPE: {self.parse.type}
        MEDICAL SPECIALTY: {self.parse.medical_specialty}
        MEDICAL PROFESSIONAL NAME: {self.parse.medical_professional_name}
        MEDICAL CONDITIONS: {self.parse.medical_conditions}
        MEDICAL INSTITUTION NAME: {self.parse.medical_institution_name}
        SUMMARY: {self.parse.summary}
        CONTENT: {self.parse.clean_content}
        """

    def __len__(self):
        return len(self.pages)

    def metadata(self):
        return {
            "filepath": self.filepath or "",
            "date": self.date or "",
            "type": str(self.parse.type).lstrip("Type.") or "",
            "medical_specialty": self.parse.medical_specialty or "",
            "medical_professional_name": self.parse.medical_professional_name or "",
            "medical_conditions": self.parse.medical_conditions or [],
            "medical_institution_name": self.parse.medical_institution_name or "",
            "summary": self.parse.summary or "",
            "clean_content": self.parse.clean_content or "",
        }
