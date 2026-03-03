import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_children(elem, name: str):
    return [c for c in list(elem) if strip_ns(c.tag) == name]


def find_first(elem, name: str):
    for c in list(elem):
        if strip_ns(c.tag) == name:
            return c
    return None


def text_of(elem) -> str | None:
    if elem is None or elem.text is None:
        return None
    txt = elem.text.strip()
    return txt if txt else None


def child_text(elem, name: str) -> str | None:
    return text_of(find_first(elem, name))


def count_nested_items(parent, child_name: str) -> int:
    if parent is None:
        return 0
    return sum(1 for c in list(parent) if strip_ns(c.tag) == child_name)


def safe_sample(lst, k=10):
    return lst[:k]


def inspect_drugbank_xml(input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_path)
    root = tree.getroot()

    # Usually DrugBank root contains many <drug> elements
    drugs = [c for c in list(root) if strip_ns(c.tag) == "drug"]

    stats = {
        "input_file": str(input_path),
        "root_tag": strip_ns(root.tag),
        "drugs_total": len(drugs),
    }

    drug_type_counter = Counter()
    group_counter = Counter()
    category_counter = Counter()
    pathway_counter = Counter()
    target_count_dist = Counter()
    enzyme_count_dist = Counter()
    transporter_count_dist = Counter()
    carrier_count_dist = Counter()
    interaction_count_dist = Counter()
    product_count_dist = Counter()
    synonym_count_dist = Counter()
    brand_count_dist = Counter()
    external_identifier_resource_counter = Counter()

    sample_drugs = []
    drugs_with_targets = 0
    drugs_with_pathways = 0
    drugs_with_interactions = 0
    drugs_with_products = 0
    drugs_with_categories = 0
    drugs_with_synonyms = 0
    drugs_with_brands = 0

    total_targets = 0
    total_enzymes = 0
    total_transporters = 0
    total_carriers = 0
    total_interactions = 0
    total_products = 0
    total_synonyms = 0
    total_brands = 0
    total_categories = 0
    total_pathways = 0

    for drug in drugs:
        drugbank_id = None
        name = child_text(drug, "name")
        description = child_text(drug, "description")

        # DrugBank ID: usually many <drugbank-id>, one primary=true
        dbids = find_children(drug, "drugbank-id")
        for dbid in dbids:
            if dbid.attrib.get("primary") == "true":
                drugbank_id = text_of(dbid)
                break
        if drugbank_id is None and dbids:
            drugbank_id = text_of(dbids[0])

        dtype = drug.attrib.get("type")
        if dtype:
            drug_type_counter[dtype] += 1

        # groups
        groups_parent = find_first(drug, "groups")
        groups = [text_of(x) for x in find_children(groups_parent, "group")] if groups_parent is not None else []
        for g in groups:
            if g:
                group_counter[g] += 1

        # categories
        categories_parent = find_first(drug, "categories")
        categories = []
        if categories_parent is not None:
            for cat in find_children(categories_parent, "category"):
                cat_name = child_text(cat, "category")
                if cat_name:
                    categories.append(cat_name)
                    category_counter[cat_name] += 1

        # pathways
        pathways_parent = find_first(drug, "pathways")
        pathway_count = 0
        if pathways_parent is not None:
            for p in find_children(pathways_parent, "pathway"):
                pathway_name = child_text(p, "name")
                if pathway_name:
                    pathway_counter[pathway_name] += 1
                pathway_count += 1

        # targets / enzymes / transporters / carriers
        targets_parent = find_first(drug, "targets")
        enzymes_parent = find_first(drug, "enzymes")
        transporters_parent = find_first(drug, "transporters")
        carriers_parent = find_first(drug, "carriers")

        target_count = count_nested_items(targets_parent, "target")
        enzyme_count = count_nested_items(enzymes_parent, "enzyme")
        transporter_count = count_nested_items(transporters_parent, "transporter")
        carrier_count = count_nested_items(carriers_parent, "carrier")

        # interactions
        interactions_parent = find_first(drug, "drug-interactions")
        interaction_count = count_nested_items(interactions_parent, "drug-interaction")

        # products / synonyms / brands
        products_parent = find_first(drug, "products")
        synonyms_parent = find_first(drug, "synonyms")
        brands_parent = find_first(drug, "international-brands")

        product_count = count_nested_items(products_parent, "product")
        synonym_count = count_nested_items(synonyms_parent, "synonym")
        brand_count = count_nested_items(brands_parent, "international-brand")

        # external identifiers
        ext_ids_parent = find_first(drug, "external-identifiers")
        if ext_ids_parent is not None:
            for ext in find_children(ext_ids_parent, "external-identifier"):
                resource = child_text(ext, "resource")
                if resource:
                    external_identifier_resource_counter[resource] += 1

        # accumulate
        target_count_dist[target_count] += 1
        enzyme_count_dist[enzyme_count] += 1
        transporter_count_dist[transporter_count] += 1
        carrier_count_dist[carrier_count] += 1
        interaction_count_dist[interaction_count] += 1
        product_count_dist[product_count] += 1
        synonym_count_dist[synonym_count] += 1
        brand_count_dist[brand_count] += 1

        total_targets += target_count
        total_enzymes += enzyme_count
        total_transporters += transporter_count
        total_carriers += carrier_count
        total_interactions += interaction_count
        total_products += product_count
        total_synonyms += synonym_count
        total_brands += brand_count
        total_categories += len(categories)
        total_pathways += pathway_count

        if target_count > 0:
            drugs_with_targets += 1
        if pathway_count > 0:
            drugs_with_pathways += 1
        if interaction_count > 0:
            drugs_with_interactions += 1
        if product_count > 0:
            drugs_with_products += 1
        if len(categories) > 0:
            drugs_with_categories += 1
        if synonym_count > 0:
            drugs_with_synonyms += 1
        if brand_count > 0:
            drugs_with_brands += 1

        if len(sample_drugs) < 15:
            sample_drugs.append({
                "drugbank_id": drugbank_id,
                "name": name,
                "type": dtype,
                "groups": groups[:5],
                "categories_count": len(categories),
                "pathways_count": pathway_count,
                "targets_count": target_count,
                "enzymes_count": enzyme_count,
                "transporters_count": transporter_count,
                "carriers_count": carrier_count,
                "interactions_count": interaction_count,
                "products_count": product_count,
                "synonyms_count": synonym_count,
                "brands_count": brand_count,
                "has_description": description is not None,
            })

    result = {
        "input_file": str(input_path),
        "root_tag": strip_ns(root.tag),
        "drugs_total": len(drugs),
        "drug_type_counts": dict(drug_type_counter.most_common()),
        "top_groups": [{"item": k, "count": v} for k, v in group_counter.most_common(30)],
        "top_categories": [{"item": k, "count": v} for k, v in category_counter.most_common(30)],
        "top_pathways": [{"item": k, "count": v} for k, v in pathway_counter.most_common(30)],
        "top_external_identifier_resources": [
            {"item": k, "count": v} for k, v in external_identifier_resource_counter.most_common(30)
        ],
        "aggregate_counts": {
            "total_targets": total_targets,
            "total_enzymes": total_enzymes,
            "total_transporters": total_transporters,
            "total_carriers": total_carriers,
            "total_interactions": total_interactions,
            "total_products": total_products,
            "total_synonyms": total_synonyms,
            "total_brands": total_brands,
            "total_categories": total_categories,
            "total_pathways": total_pathways,
        },
        "coverage": {
            "drugs_with_targets": drugs_with_targets,
            "drugs_with_pathways": drugs_with_pathways,
            "drugs_with_interactions": drugs_with_interactions,
            "drugs_with_products": drugs_with_products,
            "drugs_with_categories": drugs_with_categories,
            "drugs_with_synonyms": drugs_with_synonyms,
            "drugs_with_brands": drugs_with_brands,
        },
        "distributions": {
            "targets_per_drug_top": [{"item": k, "count": v} for k, v in target_count_dist.most_common(20)],
            "enzymes_per_drug_top": [{"item": k, "count": v} for k, v in enzyme_count_dist.most_common(20)],
            "transporters_per_drug_top": [{"item": k, "count": v} for k, v in transporter_count_dist.most_common(20)],
            "carriers_per_drug_top": [{"item": k, "count": v} for k, v in carrier_count_dist.most_common(20)],
            "interactions_per_drug_top": [{"item": k, "count": v} for k, v in interaction_count_dist.most_common(20)],
        },
        "sample_drugs": sample_drugs,
        "recommendation_hints": {
            "candidate_for_entity_relation_extraction": len(drugs) > 0,
            "has_rich_biomedical_relations": total_targets + total_enzymes + total_transporters + total_carriers > 0,
            "has_many_textual_fields": total_synonyms + total_products + total_brands > 0,
            "likely_needs_dataset_specific_preprocess": True,
        },
    }

    with (output_dir / "drugbank_xml_inspection.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "DrugBank XML Inspection Summary",
        "=" * 36,
        f"Input file: {input_path}",
        f"Root tag: {strip_ns(root.tag)}",
        f"Drugs total: {len(drugs)}",
        "",
        "Drug type counts:",
    ]
    for k, v in drug_type_counter.most_common(10):
        summary_lines.append(f"  - {k}: {v}")

    summary_lines.extend([
        "",
        "Coverage:",
        f"  - drugs with targets: {drugs_with_targets}",
        f"  - drugs with pathways: {drugs_with_pathways}",
        f"  - drugs with interactions: {drugs_with_interactions}",
        f"  - drugs with categories: {drugs_with_categories}",
        f"  - drugs with synonyms: {drugs_with_synonyms}",
        "",
        "Aggregate counts:",
        f"  - total targets: {total_targets}",
        f"  - total enzymes: {total_enzymes}",
        f"  - total transporters: {total_transporters}",
        f"  - total carriers: {total_carriers}",
        f"  - total interactions: {total_interactions}",
        "",
        "Hints:",
        f"  - candidate_for_entity_relation_extraction: {result['recommendation_hints']['candidate_for_entity_relation_extraction']}",
        f"  - has_rich_biomedical_relations: {result['recommendation_hints']['has_rich_biomedical_relations']}",
        f"  - has_many_textual_fields: {result['recommendation_hints']['has_many_textual_fields']}",
        f"  - likely_needs_dataset_specific_preprocess: {result['recommendation_hints']['likely_needs_dataset_specific_preprocess']}",
    ])

    with (output_dir / "drugbank_xml_inspection_summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect DrugBank full_database.xml before preprocessing.")
    parser.add_argument("--input", required=True, help="Path to DrugBank XML file")
    parser.add_argument("--output-dir", required=True, help="Directory for inspection outputs")
    args = parser.parse_args()

    result = inspect_drugbank_xml(Path(args.input), Path(args.output_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()