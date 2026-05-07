import subprocess
import textwrap
from pathlib import Path


def test_deferred_acl_cleanup_scope_runs_deferred_callbacks(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "cleanup_scope_test.cpp"
    source.write_text(
        textwrap.dedent(
            r"""
            #include <cassert>
            #include <vector>

            #include "ascend/graph_cleanup_.h"

            int main() {
              using infini::ops::ascend::DeferredAclCleanupScope;
              using infini::ops::ascend::DeferOrRunAclCleanup;

              std::vector<int> events;
              DeferOrRunAclCleanup([&] { events.push_back(1); });
              assert((events == std::vector<int>{1}));

              {
                DeferredAclCleanupScope scope;
                DeferOrRunAclCleanup([&] { events.push_back(2); });
                DeferOrRunAclCleanup([&] { events.push_back(3); });
                assert((events == std::vector<int>{1}));
              }
              assert((events == std::vector<int>{1, 2, 3}));

              {
                DeferredAclCleanupScope outer;
                DeferOrRunAclCleanup([&] { events.push_back(4); });
                {
                  DeferredAclCleanupScope inner;
                  DeferOrRunAclCleanup([&] { events.push_back(5); });
                }
                assert((events == std::vector<int>{1, 2, 3, 5}));
              }
              assert((events == std::vector<int>{1, 2, 3, 5, 4}));

              return 0;
            }
            """
        )
    )
    binary = tmp_path / "cleanup_scope_test"

    subprocess.run(
        ["c++", "-std=c++17", f"-I{repo / 'src'}", str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_deferred_acl_cleanup_scope_can_release_callbacks(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "cleanup_scope_release_test.cpp"
    source.write_text(
        textwrap.dedent(
            r"""
            #include <cassert>
            #include <vector>

            #include "ascend/graph_cleanup_.h"

            int main() {
              using infini::ops::ascend::DeferredAclCleanupScope;
              using infini::ops::ascend::DeferOrRunAclCleanup;

              std::vector<int> events;
              std::vector<std::function<void()>> callbacks;
              {
                DeferredAclCleanupScope scope;
                DeferOrRunAclCleanup([&] { events.push_back(1); });
                callbacks = scope.Release();
              }

              assert(events.empty());
              for (auto& callback : callbacks) {
                callback();
              }
              assert((events == std::vector<int>{1}));

              return 0;
            }
            """
        )
    )
    binary = tmp_path / "cleanup_scope_release_test"

    subprocess.run(
        ["c++", "-std=c++17", f"-I{repo / 'src'}", str(source), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
