# pyright: strict
"""
The public API in strict mode: no unknown (or partially unknown) types may leak into the user's code
"""
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch

from esorm import ESModel, ESBulk, connect, es, lazy_property, retry_on_conflict, setup_mappings
from esorm.fastapi import make_dep_sort, set_max_page_size
from esorm.fields import keyword


class Item(ESModel):
    name: keyword
    tags: List[str] = []

    @lazy_property
    async def related(self) -> List['Item']:
        return await Item.search_by_fields({'name': self.name})


@retry_on_conflict(3, reload_on_conflict=False)
async def rename(item_id: str, name: str) -> Item:
    item = await Item.get(item_id)
    item.name = name
    await item.save(wait_for=True)
    return item


async def usage() -> None:
    client: Optional[AsyncElasticsearch] = await connect('http://localhost:9200', wait=True)
    hosts: List[str] = ['http://es1:9200', 'http://es2:9200']
    await connect(hosts)
    await connect(('http://es1:9200', {'host': 'es2', 'port': 9200}))
    print(client, es)
    await setup_mappings(debug=True)
    set_max_page_size(100)

    item: Item = await rename('1', 'name')
    doc: Dict[str, Any] = item.to_es()
    res = await Item.call('get', id='1')
    print(doc, res)
    loaded: Item = Item.from_es({'_id': '1', '_source': {'name': 'name'}})
    print(loaded.name)

    async with ESBulk(wait_for=True) as bulk:
        await bulk.save(item)
        await bulk.delete(item)

    dep_sort = make_dep_sort(name_asc=[{'name': {'order': 'asc'}}])
    print(dep_sort)
