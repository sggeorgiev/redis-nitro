tags {"external:skip"} {

# Copy RDB with ziplist encoded hash to server path
set server_path [tmpdir "server.convert-ziplist-zset-on-load"]

exec cp -f tests/assets/zset-ziplist.rdb $server_path
start_server [list overrides [list "dir" $server_path "dbfilename" "zset-ziplist.rdb"]] {
    test "RDB load ziplist zset: converts to btree when RDB loading" {
        r select 0

        assert_encoding btree zset
        assert_equal 2 [r zcard zset]
        assert_match {one 1 two 2} [r zrange zset 0 -1 withscores]
    }
}

}
