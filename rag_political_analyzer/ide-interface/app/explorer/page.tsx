import DataSourceExplorer from "@/app/components/DataSourceExplorer";

export default function ExplorerPage() {
  return (
    <div>
      <h1 className="text-xl font-semibold mb-4 p-4 border-b border-gray-700">Data Source Explorer</h1>
      <div className="p-4">
        <p className="text-sm text-gray-400 mb-4">
          This view shows all the root URLs and documents that have been ingested into the system.
          You can manage them from the sidebar explorer on the left.
        </p>
        <p className="text-sm text-gray-400">
            In a future version, clicking on a data source could reveal all its indexed text and code chunks right here in the main editor pane.
        </p>
      </div>
    </div>
  );
}
