This will install redis in the cluster, and expose it via the existing MetalLB pool.

to install it, modify the `values.yaml` file here, then run:
```
helm install redis-langchain oci://registry-1.docker.io/bitnamicharts/redis --namespace <your namespace> -f values.yaml
```

Since this is a standalone, memory-only instance of redis, there isn't any need for storage classes or authentication
the way there normally is for redis. What you *do* need to do, though, is set up the authentication for access
to redis itself, with the `secrets.yaml` file. (don't commit that to git, though)