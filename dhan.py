def hierarchical(points, k):
    h = []
    valid = set([str(x) for x in range(len(points))])
    entry_order = {str(x): x for x in range(len(points))}
    centroids = {}
    for x in range(len(points)):
        centroids[str(x)] = points[x]
    for combo in itertools.combinations(range(len(points)), 2):
        h.append([eucledian_distance(points[combo[0]], points[combo[1]]), combo[0], [
                 str(combo[0]), str(combo[1])]])
    heapq.heapify(h)
    _ = 0
    while _ < len(points)-k:
        #print(h)
        c = heapq.heappop(h)
        invalid = False
        for cluster_name in c[2]:
            if cluster_name not in valid:
                invalid = True
                break
        if invalid:
            continue
        #print("merging::::::::::"+c[2][0]+"---"+c[2][1]+"\t"+str(c[0])+"\n")
        _ += 1
        new_cluster_name = ",".join(c[2])
        entry_order[new_cluster_name] = min(
            entry_order[c[2][0]], entry_order[c[2][1]])
        for cluster_name in c[2]:
            valid.remove(cluster_name)
        #h = eager_clean(c[2], h)
        centroids[new_cluster_name] = get_centroid(points, new_cluster_name)
        for valid_cluster in valid:
            h.append([eucledian_distance(centroids[valid_cluster], centroids[new_cluster_name]), min(entry_order[valid_cluster], entry_order[new_cluster_name]), [
                valid_cluster, new_cluster_name]])
        valid.add(new_cluster_name)
        heapq.heapify(h)
    return valid, centroids