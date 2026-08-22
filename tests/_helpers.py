from regression_records.serialization import from_json_tree, to_json_tree


def round_trip(obj, ctx_pair):
    ser_ctx, deser_ctx = ctx_pair
    tree = to_json_tree(obj, ser_ctx)
    return from_json_tree(tree, deser_ctx)
