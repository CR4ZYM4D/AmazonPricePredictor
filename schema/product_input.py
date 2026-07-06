from pydantic import BaseModel
from typing import Optional

class ProductInput(BaseModel):
    sample_id: Optional[int]
    catalog_content: str
    image_link: Optional[str]



