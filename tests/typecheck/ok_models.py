# pyright: standard
"""
Correct model definitions and usage, type checkers must not report anything here
"""
from typing import Optional, List
from datetime import datetime

from esorm import ESModelTimestamp, ESBaseModel, ESBulk, Pagination, Sort, lazy_property, retry_on_conflict
from esorm.fields import Field, TextField, NumericField, DenseVectorField, keyword, text, dense_vector, long


class Address(ESBaseModel):
    city: str
    zip: keyword


class User(ESModelTimestamp):
    class ESConfig:
        index = 'users'
        id_field = 'email'
        default_sort = [{'name': {'order': 'asc'}}]

    email: keyword
    name: str = TextField(..., min_length=3, max_length=100)
    title: text = TextField(..., keyword=True)
    nick: Optional[str] = TextField(None)
    description: str = TextField(description="The description of the user")
    price: float = NumericField(..., gt=0)
    age: int = NumericField(18)
    points: long = NumericField(0, ge=0)
    is_available: bool = Field(True)
    full_name: keyword = Field(alias='fullName')
    tags: List[str] = Field([], index=False)
    embedding: dense_vector = DenseVectorField(..., dims=3, similarity="cosine")
    address: Optional[Address] = None
    friend_ids: List[str] = []
    logged_in_at: Optional[datetime] = None

    @lazy_property
    async def friends(self) -> List['User']:
        return await User.search({'ids': {'values': self.friend_ids}})


async def crud() -> None:
    user = User(email='a@b.c', name='John', title='Mr', description='x', price=1.0, fullName='John Doe',
                embedding=[0.1, 0.2, 0.3], address=Address(city='Budapest', zip='1111'))
    doc_id: str = await user.save()
    user2 = await User.get(doc_id)
    print(user2.name, user2.email, user2.created_at, user2.modified_at)
    if user2.address:
        print(user2.address.city)
    user3: User = await user2.reload()
    print(user3.name)
    user3.age += 1
    await user3.save(refresh='wait_for')
    await user3.delete()
    print(User.ESConfig.index)


async def decorators() -> None:
    pagination = Pagination(page=1, page_size=10)
    sort = Sort(sort=[{'name': {'order': 'desc'}}])
    users = await pagination(sort(User)).search({'match_all': {}})
    for user in users:
        print(user.name)


async def bulk() -> None:
    async with ESBulk() as b:
        user = await User.get('x@y.z')
        await b.save(user)
        await b.delete(user)


@retry_on_conflict(3)
async def update_age(email: str) -> None:
    user = await User.get(email)
    user.age += 1
    await user.save()
