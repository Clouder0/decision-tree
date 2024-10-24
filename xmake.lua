set_languages("c++23")
set_optimize("fast")
add_vectorexts("avx2")


target("tree")
    set_kind("shared")
    add_files("src/*.cc")
    add_includedirs("include")
